# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import matplotlib.pyplot as plt

from typing import Any, Dict, List, Optional, Tuple

from modeling.sam import Sam
from sam_predictor import SamPredictor
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class MaskDataBP(MaskData):
    """
    Removes the deepcopy operation from the cat method to allow for backpropagation.
    """
    def cat(self, new_stats: "MaskData") -> None:
        for k, v in new_stats.items():
            if k not in self._stats or self._stats[k] is None:
                self._stats[k] = v
            elif isinstance(v, torch.Tensor):
                self._stats[k] = torch.cat([self._stats[k], v], dim=0)
            elif isinstance(v, np.ndarray):
                self._stats[k] = np.concatenate([self._stats[k], v], axis=0)
            elif isinstance(v, list):
                self._stats[k] = self._stats[k] + v
            else:
                raise TypeError(f"MaskData key {k} has an unsupported type {type(v)}.")


class SamPointMaskGenerator():
    """
    Using a SAM model, generates masks from the given points and labels
    with the same NMS and stability score filtering as SAM prediction with a point grid
    for the whole image.
    So basically SAM entity segmentation where the point grid is replaced
    with the given points and labels.
    Also returns raw masks.
    Crop functionality is not supported.
    """
    def __init__(
        self,
        model: Sam,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode

    # @torch.no_grad()
    def generate(
            self,
            image: np.ndarray,
            input_points: np.ndarray,
            input_labels: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
               bbox (list(float)): The box around the mask, in XYWH format.
               area (int): The area in pixels of the mask.
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
        """
        # make sure points are B x N x 2
        assert input_points.ndim == 3 and input_points.shape[2] == 2, "input_points must have shape B x N x 2"

        # Generate masks
        mask_data = self._generate_masks(image, input_points, input_labels)

        # Filter small disconnected regions and holes in masks
        # if self.min_mask_region_area > 0:
        #     mask_data = self.postprocess_small_regions(
        #         mask_data,
        #         self.min_mask_region_area,
        #         self.box_nms_thresh,
        #     )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "segmentation_raw": torch.sigmoid(mask_data["masks_raw"][idx]),
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx],
                "point_coords": [mask_data["points"][idx].tolist()],
                "point_idx": [mask_data["point_idxs"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(
            self,
            image: np.ndarray,
            input_points: np.ndarray,
            input_labels: np.ndarray
    ) -> MaskDataBP:
        orig_size = image.shape[:2]

        crop_boxes = self.generate_im_crop_box(orig_size)

        # Iterate over image crops
        data = MaskDataBP()

        for crop_box in crop_boxes:
            crop_data = self._process_crop(image, crop_box, orig_size, input_points, input_labels)
            data.cat(crop_data)

        #     we assume there are no crops, just the full image, so no need for NMS etc here
        # data.to_numpy()
        return data

    def generate_im_crop_box(self, im_size: Tuple[int, ...]) -> Tuple[List[int], List[int]]:
        im_h, im_w = im_size

        # Original image
        crop_boxes = [[0, 0, im_w, im_h]]

        return crop_boxes

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        input_points: np.ndarray,
        input_labels: np.ndarray
    ) -> MaskDataBP:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = input_points * points_scale

        # Generate masks for this crop in batches
        data = MaskDataBP()
        for batch in batch_iterator(self.points_per_batch, points_for_image, input_labels):
            points, labels = batch
            batch_data = self._process_batch(points, labels, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # # Remove duplicates within this crop.
        # keep_by_nms = batched_nms(
        #     data["boxes"].float(),
        #     data["iou_preds"],
        #     torch.zeros_like(data["boxes"][:, 0]),  # categories
        #     iou_threshold=self.box_nms_thresh,
        # )
        # data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskDataBP:
        orig_h, orig_w = orig_size

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.tensor(labels, dtype=torch.int, device=in_points.device)
        # TODO check what happens to multimask
        masks, iou_preds, _ = self.predictor.predict_torch(
            in_points,
            in_labels,
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        # data = MaskDataBP(
        #     masks=masks.flatten(0, 1),
        #     iou_preds=iou_preds.flatten(0, 1),
        #     points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        # )

        # only pick the first (biggest) out of the three masks for each point
        data = MaskDataBP(
            masks=masks[:, 0, :, :],
            iou_preds=iou_preds[:, 0],
            points=torch.as_tensor(points),
            point_idxs=torch.arange(len(points))
        )

        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks_raw"] = data["masks"]
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries - only for inner crops, which are not used here?
        # keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        # if not torch.all(keep_mask):
        #     data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["masks_raw"] = uncrop_masks(data["masks_raw"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskDataBP, min_area: int, nms_thresh: float
    ) -> MaskDataBP:
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:
            mask = rle_to_mask(rle)

            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(masks)
        keep_by_nms = batched_nms(
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms:
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly
        mask_data.filter(keep_by_nms)

        return mask_data

    def train_norm_layers_only(self):
        self.predictor.model.train_norm_layers_only()

    def freeze_seg_decoder(self):
        self.predictor.model.freeze_seg_decoder()

    def freeze_encoder(self):
        self.predictor.model.freeze_encoder()

    def freeze_prompt_encoder(self):
        self.predictor.model.freeze_prompt_encoder()

    def eval(self):
        self.predictor.model.eval()

    def get_adv_gt(self, gt, mask, thresh=0.4):
        if gt == 'invert':
            gt = (mask < thresh).float()
        if gt == 'random':
            # generate random gt
            r = (torch.rand_like(mask) > thresh).float()
            # blur gt
            r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
            gt = mask.clone()
            # gt[r > 0] = r[r > 0].detach()
            gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

        return gt

    def fgsm_attack(self, img, pts, labels, epsilon=0.05, debug=False, gt='invert'):
        assert gt in ['invert', 'random']
        self.predictor.set_adversarial_image(img)
        masks, iou_preds, _ = self.predictor.predict_torch(pts, labels, multimask_output=True, return_logits=True)
        # take the biggest prediction
        mask = torch.sigmoid(masks[:, 0, :, :])

        gt = self.get_adv_gt(gt, mask)

        loss = torch.nn.functional.binary_cross_entropy(mask, gt)

        # zero image gradients
        self.predictor.input_image.grad = None
        # Calculate gradients of model in backward pass
        loss.backward()

        data_grad = self.predictor.input_image.grad.data

        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_images = self.predictor.input_image + epsilon * sign_data_grad

        self.predictor.reset_image()

        # includes clipping
        adv_im = self.predictor.tensor_image_to_numpy(perturbed_images, (img.shape[0], img.shape[1]))
        return adv_im

    def pgd_attack(self, img,  pts, labels, iters=40, lr=0.005, epsilon=0.05, debug=False, gt='invert'):
        assert gt in ['invert', 'random']

        gt_name = gt

        self.predictor.set_adversarial_image(img)

        # the normalized and resized image used by SAM
        orig_img_tensor = self.predictor.input_image.clone().detach()

        optim = torch.optim.Adam([self.predictor.input_image], lr=lr)

        if debug:
            fig, ax = plt.subplots(2, int(np.ceil((iters // 2))), figsize=(11, 4))
            #     ticks off for all axes
            for a in ax.flatten():
                a.set_xticks([])
                a.set_yticks([])
        for it in range(iters):
            if it > 0:
                # recompute image features from updated img tensor
                self.predictor.features = self.predictor.model.image_encoder(self.predictor.input_image)

            # Forward pass the data through the model
            masks, iou_preds, _ = self.predictor.predict_torch(pts, labels, multimask_output=True, return_logits=True)
            # take the biggest prediction
            mask = torch.sigmoid(masks[:, 0, :, :])

            # show mask
            if debug and (it % 2 == 0 or it == iters - 1):
                ax[1][it // 2].imshow(mask.detach().cpu().numpy().squeeze())
                ax[0][it // 2].imshow(self.predictor.tensor_image_to_numpy(self.predictor.input_image,
                                                                           (img.shape[0], img.shape[1])).squeeze())
                # plt.title(f'mask {it}')
                # plt.show()
            if it == 0:
                gt = self.get_adv_gt(gt, mask)
                #     otherwise backward fails since we pass it multiple times
                gt = gt.detach()
                # if debug:
                #     plt.imshow(gt.detach().cpu().numpy().squeeze())
                #     plt.title(f'gt')
                #     plt.show()

                seg_preds_orig = mask.clone()

            # Calculate the loss
            loss = torch.nn.functional.binary_cross_entropy_with_logits(mask, gt)

            # Zero all existing gradients
            optim.zero_grad()

            # retain_graph since encoder features are reused
            torch.autograd.set_detect_anomaly(True)

            loss.backward()

            # perform optimization step
            optim.step()

            # clipping to max epsilon difference
            delta = self.predictor.input_image.detach() - orig_img_tensor
            delta_norm = torch.abs(delta)
            div = torch.clamp(delta_norm / epsilon, min=1.0)
            delta = delta / div
            self.predictor.input_image.data = orig_img_tensor + delta

        ims_adv_tensor = self.predictor.tensor_image_to_numpy(self.predictor.input_image,  (img.shape[0], img.shape[1]))

        self.predictor.reset_image()
        #make spaces between images smaller
        plt.tight_layout()
        plt.show()

        # also show input, gt and original mask
        if debug:
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # all ticks off
            for a in ax.flatten():
                a.set_xticks([])
                a.set_yticks([])
            ax[0].imshow(img)
            ax[0].set_title('input', fontsize=20)
            ax[1].imshow(seg_preds_orig.detach().cpu().numpy().squeeze())
            ax[1].set_title('original mask', fontsize=20)
            ax[2].imshow(gt.detach().cpu().numpy().squeeze())
            ax[2].set_title(f'{gt_name} gt', fontsize=20)
            plt.tight_layout()
            plt.show()
        return ims_adv_tensor