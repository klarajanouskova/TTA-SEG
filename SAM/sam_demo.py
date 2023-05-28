import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

from segment_anything import SamAutomaticMaskGenerator
from sam_predictor import SamPredictor
from modeling.build_sam import sam_model_registry

device = "cuda" if torch.cuda.is_available() else "cpu"


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def get_sam_entity_segmenter():
    """
    Entity segmentation model
    """
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam = sam.to(device)

    # full image
    mask_generator = SamAutomaticMaskGenerator(sam,
                                               points_per_side=10,
                                               pred_iou_thresh=0.86,
                                               stability_score_thresh=0.92,
                                               crop_n_layers=1,
                                               crop_n_points_downscale_factor=2,
                                               min_mask_region_area=100)
    return mask_generator


def get_sam_predictor():
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    return predictor


def run_entitiy_segmentation(img, sam_gen, vis=False):
    """
    Run entity segmentation on a single image
    """
    masks = sam_gen.generate(np.array(img))
    if vis:
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_anns(masks)
        plt.axis('off')
        plt.show()

    return masks


def run_point_instance_segmentation(img, sam_pred, input_points, input_labels, vis=False):
    sam_pred.set_image(img)


    # batched version: masks, _, _ = predictor.predict_torch(
    #     point_coords=None,
    #     point_labels=None,
    #     boxes=transformed_boxes,
    #     multimask_output=False,
    # )
    masks, scores, logits = sam_pred.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    if vis:
        # show input points
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        show_points(input_points, input_labels, plt.gca())
        plt.axis('on')
        plt.show()

        for i, (mask, logit_mask, score) in enumerate(zip(masks, logits, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            show_mask(mask, plt.gca())
            show_points(input_points, input_labels, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

            plt.figure(figsize=(10, 10))
            plt.imshow(torch.sigmoid(torch.Tensor(logit_mask)))
            plt.title(f"Logit mask {i + 1}", fontsize=18)
            plt.axis('off')
            plt.show()

    print(scores)
    # TODO filter for biggest mask
    return masks, scores, logits


def entity_seg_demo():
    """
    Entity segmentation demo
    """
    sam_gen = get_sam_entity_segmenter()

    img = cv2.imread('../animals/cica.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to 1/4
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    # run entity segmentation
    masks = run_entitiy_segmentation(img, sam_gen, vis=True)

    print()


def point_instance_seg_demo():
    """
    Point instance segmentation demo
    """
    sam_pred = get_sam_predictor()

    img = cv2.imread('../animals/cica.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize to 1/4
    img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    # prepare points.
    # Points are input to the model in (x,y) format and come with labels 1 (foreground point) or 0 (background point).
    # Multiple points can be input; here we use only one. The chosen point will be shown as a star on the image.
    h, w = img.shape[:2]
    input_points = np.array([[w // 2, h // 2], [40, 40]])
    # center is fg, top-left is bg
    input_labels = np.array([1, 0])

    # run entity segmentation
    masks = run_point_instance_segmentation(img, sam_pred, input_points, input_labels,  vis=True)

    print()


def batched_input():
    from segment_anything.utils.transforms import ResizeLongestSide

    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b.pth")
    sam = sam.to(device)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    def prepare_image(image, transform, device):
        image = transform.apply_image(image)
        image = torch.as_tensor(image, device=device.device)
        return image.permute(2, 0, 1).contiguous()

    image1, image2 = None, None
    image1_boxes, image2_boxes = None, None
    batched_input = [
        {
            'image': prepare_image(image1, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image1_boxes, image1.shape[:2]),
            'original_size': image1.shape[:2]
        },
        {
            'image': prepare_image(image2, resize_transform, sam),
            'boxes': resize_transform.apply_boxes_torch(image2_boxes, image2.shape[:2]),
            'original_size': image2.shape[:2]
        }
    ]
    # The output is a list over results for each input image, where list elements are dictionaries with the following keys:
    #
    # masks: A batched torch tensor of predicted binary masks, the size of the original image.
    # iou_predictions: The model's prediction of the quality for each mask.
    # low_res_logits: Low res logits for each mask, which can be passed back to the model as mask input on a later iteration.
    batched_output = sam(batched_input, multimask_output=False)

    fig, ax = plt.subplots(1, 2, figsize=(20, 20))

    ax[0].imshow(image1)
    for mask in batched_output[0]['masks']:
        show_mask(mask.cpu().numpy(), ax[0], random_color=True)
    for box in image1_boxes:
        show_box(box.cpu().numpy(), ax[0])
    ax[0].axis('off')

    ax[1].imshow(image2)
    for mask in batched_output[1]['masks']:
        show_mask(mask.cpu().numpy(), ax[1], random_color=True)
    for box in image2_boxes:
        show_box(box.cpu().numpy(), ax[1])
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    point_instance_seg_demo()
