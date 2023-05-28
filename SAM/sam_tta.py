
import cv2

import models_loss

import sys
import os

import torch
from torchvision.transforms.v2.functional import to_pil_image, to_tensor

import numpy as np
import matplotlib.pyplot as plt
# prevent matpltolib form using scientific notation
plt.rcParams['axes.formatter.useoffset'] = False

from tqdm import tqdm
from util.datasets_seg import get_pascal

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from eval import load_seg_model, load_tta_model, iou_loss

from distortion import distortions, fgsm, pgd
from util.voc_dataset_seg import CA_CLEAN_IDXS_VAL_A
from eval_segmentation import load_sam
from arg_composition import get_segmentation_args

from pynvml import nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlInit

sys.path.append('..')

import gc

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


def print_cuda_mem_usage(id=0):
    if not local:
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(id)
        mem = nvmlDeviceGetMemoryInfo(handle)
        unit_scale = 1024 ** 3
        used = mem.used / unit_scale
        total = mem.total / unit_scale
        print(f'Used: {used:.2f}GB / {total:.2f}GB')


def print_tensors_on_cuda():
    """
    The output is too messy to be of any use, find sth better
    """
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except:
            pass


def plot_results_image(name, img, gt, preds_seg, preds_rec, seg_iou_losses, loss_dict, thresh=0.4,
                       save_name=None, folder=''):
    tta_method = '&'.join(loss_dict.keys())
    # get the other losses aggregation from loss dict, convert to numpy, scale when possible
    seg_iou_losses = np.array(seg_iou_losses) * 100
    tta_losses = np.array([l for l in loss_dict.values()]).sum(0)

    vis_im = im_to_vis(img)
    it_axis = np.arange(len(seg_iou_losses))

    r, c = len(preds_seg) + 1, 3
    plt.subplots(nrows=r, ncols=c, figsize=(c * 4, r * 4))

    baseline_pred = (preds_seg[0].squeeze() > thresh).int()

    for i, (pred_seg, pred_rec) in enumerate(zip(preds_seg, preds_rec)):
        pred_seg = pred_seg.squeeze()
        pred_seg_t = (pred_seg > thresh).int()
        pixel_idxs_better = np.where((pred_seg_t == gt) & (baseline_pred != gt))
        pixel_idxs_worse = np.where((pred_seg_t != gt) & (baseline_pred == gt))
        pred_seg_error_vis = pred_seg_t[:, :, None].numpy().repeat(3, axis=2)
        # make improved pixels green
        pred_seg_error_vis[pixel_idxs_better] = [0, 1, 0]
        # make worsened pixels red
        pred_seg_error_vis[pixel_idxs_worse] = [1, 0, 0]

        pred_seg_error_vis = pred_seg_error_vis * 255

        tta_impr = (tta_losses[i] - tta_losses[0]) / tta_losses[0] if i > 0 else 1
        seg_impr = (seg_iou_losses[i] - seg_iou_losses[0]) / seg_iou_losses[0] if i > 0 else 1

        # seg
        plt.subplot(r, c, i * c + 1)
        plt.imshow(pred_seg_error_vis)
        color = 'red' if seg_impr > 0 else 'green'
        if seg_impr == 0:
            color = 'black'
        plt.title(f'{i} TTA: {seg_iou_losses[i]:.3f}, it0: {seg_impr:.3f}', color=color)
        plt.axis('off')

        # raw seg pred
        plt.subplot(r, c, i * c + 2)
        plt.imshow(pred_seg)
        plt.title('Segmentation prediction')
        plt.axis('off')

        # rec
        plt.subplot(r, c, i * c + 3)
        plt.imshow(im_to_vis(pred_rec))
        color = 'red' if tta_impr > 0 else 'green'
        if tta_impr == 0:
            color = 'black'
        plt.title(f'{i} TTA: {tta_losses[i]:.3f}, it0: {tta_impr:.3f}', color=color)
        plt.axis('off')

    # raw seg pred
    plt.subplot(r, c, (i + 1) * c + 1)
    plt.imshow(vis_im)
    plt.title('Input image')
    plt.axis('off')

    plt.subplot(r, c, (i + 1) * c + 2)
    plt.imshow(gt, cmap='gray')
    plt.title(f'Ground truth')
    plt.axis('off')

    plt.subplot(r, c, (i + 1) * c + 3)
    plt.plot(it_axis, (seg_iou_losses - seg_iou_losses[0]) / seg_iou_losses[0], label='Segmentation MRE')
    plt.plot(it_axis, (tta_losses - tta_losses[0]) / tta_losses[0], label=f'TTA - {tta_method}')
    for loss_name, losses_vals in loss_dict.items():
        losses_vals = np.array(losses_vals)
        plt.plot(it_axis, (losses_vals - losses_vals[0]) / losses_vals[0], label=f'TTA - {loss_name}')
    plt.xticks(it_axis)
    plt.title('Mean Relative Error over SSL iterations')
    plt.xlabel('SSL iteration')
    plt.ylabel('MRE')
    plt.legend()

    # make dir if it doesn't exist
    os.makedirs(f'pascal_tta/{folder}', exist_ok=True)
    if save_name is not None:
        plt.savefig(f'pascal_tta/{folder}{name}_{save_name}.jpg')

    plt.show()


class TestTimeAdaptor():
    # should contain consistency-based pseudo-labelling as well
    valid_methods = ['tent', 'sam-iou', 'iou', 'ref']
    # TODO add method sorting for consistent save_name

    def __init__(self, tta_method, args, weights=None, mask_ratio=0.75, load_iou=True):
        self.tta_methods = self.verify_method(tta_method)
        if weights is None:
            weights = [1] * len(self.tta_methods)
        self.weights = weights
        assert len(self.tta_methods) == len(self.weights), 'Number of methods and weights must be the same'
        # TODO not used yet
        self.mask_ratio = mask_ratio

        self.mask_generator = None

        self.args = args

        if 'iou' in self.tta_methods or load_iou:
            args.tta_model_run = args.tta_iou_model_run
            # cityscapes input size
            self.loss_model = load_tta_model(args, models_loss.MaskLossNet, size=(512, 1024))
            # should only be used for inference
            self.loss_model.eval()
            self.loss_model.freeze()
        if 'ref' in self.tta_methods or args.tta_ref_post:
            args.tta_model_run = args.tta_ref_model_run
            # cityscapes input size
            self.ref_net = load_tta_model(args, models_loss.MaskLossUnet, size=(512, 1024))
            # should only be used for inference
            self.ref_net.eval()
            self.ref_net.freeze()

        # string from weights, each weight separated by _
        weights_str = '_'.join([str(w) for w in self.weights])
        if tta_method == 'ref':
            # before it was conf 80
            self.save_name = f'ref_tta_{tta_method}90_lr_{args.tta_lr}_ws_{weights_str}' \
                             f'_its_{args.tta_iter_num}_' \
                             f'nims_{args.tta_n_ims}'
        else:
            self.save_name = f'sam_tta_{tta_method}_lr_{args.tta_lr}_ws_{weights_str}' \
                             f'_its_{args.tta_iter_num}_' \
                             f'nims_{args.tta_n_ims}'

    def __call__(self, image, points, labels):
        """
        Assumes numpy array of shape (H, W, 3) and values in [0, 255], no normalization
        """

        # reload the weights
        self.mask_generator = load_sam(self.args.model_path)

        # set eval and freeze
        self.reset_require_grad()

        preds_tta, loss_dict = self.optimize_im(
            image,
           points,
           labels,
            num_it=self.args.tta_iter_num,
            lr=self.args.tta_lr,
            # lr=1e-2,
            debug=False
        )

        # take the output from the last optimization step
        xs_tta = preds_tta[:, -1]

        return xs_tta, preds_tta, loss_dict

    def reset_require_grad(self):
        # if 'tent' not in self.tta_methods:
        if True:
            self.mask_generator.eval()  # makes sure layers like batchnorm or dropout are in eval mode - doesn't prevent backprop
        if 'tent' in self.tta_methods or 'adv' in self.tta_methods:
            # we assume tent/adv is the only method so we can freeze stuff here
            self.mask_generator.train_norm_layers_only()
        else:
            #     set requires grad
            for param in self.mask_generator.predictor.model.parameters():
                param.requires_grad = True

    def optimize_im(
            self,
            img,
            points,
            labels,
            num_it=20,
            lr=1e-3,
            debug=True,
            # only for debugging purposes
            gt=None,
            optim='sgd',
            momentum=0.9
    ):
        assert optim in ['sgd', 'adam']
        if optim == 'sgd':
            optim = torch.optim.SGD(
                lr=lr,
                params=filter(lambda p: p.requires_grad, self.mask_generator.predictor.model.parameters()),
                momentum=momentum)
        else:
            optim = torch.optim.AdamW(
                lr=lr,
                params=filter(lambda p: p.requires_grad, self.mask_generator.predictor.model.parameters()))

        preds_seg = []
        loss_dict = {tta_method: [] for tta_method in self.tta_methods}

        for i in range(num_it + 1):
            # segmentation prediction
            losses = []
            optim.zero_grad(set_to_none=True)

            out_it = self.mask_generator.generate(img, points, labels)
            out_it = out_it[:12]
            preds_seg.append(torch.stack([out['segmentation_raw'] for out in out_it]).detach().cpu())

            for tta_method, weight in zip(self.tta_methods, self.weights):
                sam_out = self.mask_generator.generate(img, points, labels)
                # do not process more than 12 images at once, discard the rest
                # TODO this is a hack, should be fixed in the future
                sam_out = sam_out[:12]
                if tta_method == 'sam-iou':
                    im_losses_iou, im_point_preds = [], []
                    # TODO aggregate to single tensor, run rec only once
                    for sam_pred in sam_out:
                        pred_mask = sam_pred['segmentation_raw']
                        #  The mask is different in each iteration so the loss values are not comparable
                        im_losses_iou += [sam_pred['predicted_iou']]
                    #     this works as long as the loss is computed for each image separately, as the model returns
                    #     mean loss by default
                    im_losses_iou = torch.stack(im_losses_iou)
                    loss_iou_agg = im_losses_iou.mean()
                    losses += [loss_iou_agg]
                    loss_dict[tta_method].append(im_losses_iou.cpu().detach().numpy())
                if tta_method == 'ref':
                    tmp_preds = []
                    # make max batch size 8
                    max_bs = 8
                    im_losses_ref = []
                    for sam_pred in sam_out:
                        pred_mask = sam_pred['segmentation_raw']
                        sam_pred['segmentation_raw'] = None
                        #  The mask is different in each iteration so the loss values are not comparable
                        tmp_preds += [pred_mask]
                    #     this works as long as the loss is computed for each image separately, as the model returns
                    #     mean loss by default
                        if len(tmp_preds) == max_bs:
                            tmp_preds = torch.stack(tmp_preds)[:, None]
                            im_losses_ref += [self.get_refinement_loss(tmp_preds, vis=False)]
                            tmp_preds = []
                    if len(tmp_preds) > 0:
                        tmp_preds = torch.stack(tmp_preds)[:, None]
                        im_losses_ref += [self.get_refinement_loss(tmp_preds, vis=False)]
                    im_losses_ref = torch.hstack(im_losses_ref)
                    loss_iou_agg = im_losses_ref.mean()
                    losses += [loss_iou_agg]
                    loss_dict[tta_method].append(im_losses_ref.cpu().detach().numpy())
                if tta_method == 'iou':
                    max_bs = 8
                    im_losses_iou = []
                    for sam_pred in sam_out:
                        pred_mask = sam_pred['segmentation_raw']
                        #  The mask is different in each iteration so the loss values are not comparable
                        tmp_preds += [pred_mask]
                        #     this works as long as the loss is computed for each image separately, as the model returns
                        #     mean loss by default
                        if len(tmp_preds) == max_bs:
                            tmp_preds = torch.stack(tmp_preds)[:, None]
                            im_losses_iou += [self.get_iout_loss(tmp_preds, vis=False)]
                            tmp_preds = []
                    if len(tmp_preds) > 0:
                        tmp_preds = torch.stack(tmp_preds)[:, None]
                        im_losses_iou += [self.get_iou_loss(tmp_preds, vis=False)]
                    im_losses_iou = torch.hstack(im_losses_iou)
                    loss_iou_agg = im_losses_iou.mean()
                    losses += [loss_iou_agg]
                    loss_dict[tta_method].append(im_losses_iou.cpu().detach().numpy())
                if tta_method == 'tent':
                    tmp_preds = []
                    for sam_pred in sam_out:
                        pred_mask = sam_pred['segmentation_raw']
                        tmp_preds += [pred_mask]
                    im_losses_tent = self.get_tent_loss(torch.stack(tmp_preds))
                    loss_seg_agg = im_losses_tent.mean()
                    # aggregate over spatial dimension, keep batch dim
                    # loss_dict[tta_method].append(loss_seg.mean(axis=(1, 2)).cpu().detach().numpy())
                    losses += [loss_seg_agg]
                    # each submask separately
                    loss_dict[tta_method].append(im_losses_tent.mean(axis=(1, 2)).cpu().detach().numpy())
                if tta_method == 'adv':
                    im_losses_adv = []
                    for instance_pts, instance_labels in zip(points, labels):
                        im_losses_adv += [self.get_adversarial_loss(img, instance_pts, instance_labels)]

                    im_losses_adv = torch.stack(im_losses_adv)
                    loss_seg_agg = im_losses_adv.mean()
                    # aggregate over spatial dimension, keep batch dim
                    # note: not mathematically correct KL
                    # losses += [loss_seg.sum() / loss_seg.shape[0]]
                    # loss_dict[tta_method].append(loss_seg.sum(axis=(1, 2, 3, 4)).cpu().detach().numpy())
                    losses += [loss_seg_agg]
                    loss_dict[tta_method].append(im_losses_adv.cpu().detach().numpy())
                sam_out = None
            loss = sum([loss * weight for loss, weight in zip(losses, self.weights)])

            if i != num_it:
                # backward on loss
                optim.zero_grad()
                loss.backward()
                optim.step()

                # free memory
                out_it, pred_seg_it = None, None    # helps to decrease gpu memory used
                gc.collect()
                torch.cuda.empty_cache()

            if debug:
                print_str = f'losses iter {i}: ' + '; '.join([f'{tta_method}: {val}' for tta_method, val in zip(self.tta_methods, losses)])
                if gt is not None:
                    iou = 1 - ((preds_seg * gt).sum() / ((preds_seg + gt)).sum())
                    print_str += f'; iou: {iou}'
                print(print_str)

        preds_seg = torch.stack(preds_seg).detach().cpu()
        # it x bs x h x w - > bs x it x h x w
        preds_seg = preds_seg.permute(1, 0, 2, 3)
        return preds_seg, loss_dict

    def verify_method(self, method):
        sub_methods = method.split('&')
        # make sure gc is not combined with other methods, as it is not supported yet
        if 'gc' in sub_methods and len(sub_methods) > 1:
            raise ValueError(f'Invalid tta method {method}, gc cannot be combined with other methods')
        for sub in sub_methods:
            if sub not in self.valid_methods:
                raise ValueError(f'Invalid tta method {sub}')
        return sub_methods

    def get_refinement_loss(self, mask_pred_tensor, return_mask=False, vis=False):
        mask_ref = self.ref_net(mask_pred_tensor)
        if vis:
            for mr, mp in zip(mask_ref, mask_pred_tensor):
                plt.subplot(1, 2, 1)
                plt.imshow(mr.squeeze().detach().cpu().numpy())
                plt.title('mask_ref')
                plt.subplot(1, 2, 2)
                plt.imshow(mp.squeeze().detach().cpu().numpy())
                plt.title('mask_pred')
                plt.show()
        # maybe this should be BCE?
        if return_mask:
            return iou_loss(mask_ref, mask_pred_tensor, apply_sigmoid=False).mean(1), mask_ref
        else:
            return iou_loss(mask_ref, mask_pred_tensor, apply_sigmoid=False).mean(1)

    def get_iou_loss(self, mask_pred_tensor):
        loss_pred = self.loss_model(mask_pred_tensor)
        return loss_pred

    def get_tent_loss(self, mask_pred_tensor):
        """
        mask_pred_tensor: tensor of shape (bs, h, w), sigmoid already applied
        """
        output_entropy = (-(torch.log(mask_pred_tensor) * mask_pred_tensor))

        return output_entropy

    def get_adversarial_loss(self, img, pts, labels):
        # we only want to optimize the image here, not the model
        self.mask_generator.freeze_seg_decoder()
        self.mask_generator.freeze_encoder()
        self.mask_generator.freeze_prompt_encoder()

        pts = torch.tensor([pts], dtype=torch.float32, device=self.mask_generator.predictor.device)
        labels = torch.tensor([labels], dtype=torch.int64, device=self.mask_generator.predictor.device)

        # scale pts by image size
        pts = pts * torch.tensor([img.shape[1], img.shape[0]], dtype=torch.float32, device=self.mask_generator.predictor.device)

        # compute adversarial images
        # get gt kind with 50 prob of each
        gt_kind = 'random' if np.random.rand() < 0.5 else 'random'
        adv_img = self.mask_generator.fgsm_attack(img, pts, labels, gt=gt_kind, debug=False)
        # adv_img = self.mask_generator.pgd_attack(img, pts, labels, gt='random', debug=True, iters=10, lr=0.0001)

        # reset grad requirements
        self.reset_require_grad()

        plt.imshow(adv_img)
        plt.show()

        plt.imshow(img)
        plt.show()

        plt.imshow(np.abs(img.astype(float) - adv_img.astype(float)).astype(np.uint8))
        plt.show()

        # compute KL loss between predictions on adversarial images and original images
        with torch.no_grad():
            # we don't want gradients here
            self.mask_generator.predictor.set_image(img)
            preds_seg = self.mask_generator.predictor.predict_torch(pts, labels, multimask_output=True, return_logits=True)[0][:, 0]
        self.mask_generator.predictor.set_image(adv_img)
        adv_preds_seg = self.mask_generator.predictor.predict_torch(pts, labels, multimask_output=True, return_logits=True)[0][:, 0]

        # compute KL loss
        loss = torch.nn.functional.kl_div(torch.stack([adv_preds_seg, 1 - adv_preds_seg], dim=-1).log(),
                                   torch.stack([preds_seg, 1 - preds_seg], dim=-1), reduction='none')
        return loss

    def forward_segmentation(self, im_tensors, inference=True):
        # if 'rec' not in self.tta_methods:
        #     raise ValueError('Reconstruction TTA method must be used to be abel to forward segmentation')
        if self.mask_generator is None:
            # lazy segmentation model loading
            self.mask_generator = load_seg_model(self.args)
            self.mask_generator.to(device)
            self.mask_generator.eval()
        with torch.no_grad():
            preds_seg = self.mask_generator.forward_seg(im_tensors, inference=inference)
        return preds_seg

    def forawrd_reifnement(self, masks_pred_tensor, vis=True):
        masks_ref = self.ref_net(masks_pred_tensor)
        if vis:
            plt.subplot(1, 2, 1)
            plt.imshow(masks_ref[0].squeeze().detach().cpu().numpy())
            plt.title('mask_ref')
            plt.subplot(1, 2, 2)
            plt.imshow(masks_pred_tensor[0].squeeze().detach().cpu().numpy())
            plt.title('mask_pred')
            plt.show()
        return masks_ref


def test_tta(args, thresh=0.4):
    args.data_cls_sub = 'dog&cow&bird'
    dataset = get_pascal(args, split='val')

    tta_methods = ['adv']
    # tta_methods = ['rec+iou']
    args.tta_grad_clip = -0.1
    # 5e-3
    args.tta_lr = 0.1
    args.tta_iter_num = 10
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[1] * len(tta_methods))

    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
        # for corr in ['glass_blur', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    for corr in ['gaussian_noise']:
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        for sev in [5]:

            # TODO handle idxs properly, sort them by TTA?
            # for idx in range(24, 30):
            for idx in CA_CLEAN_IDXS_VAL_A:
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int()

                # distort and add batch dimension
                if corrupt_fun is not None:
                    denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                                torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor = img[None].float().to(device)
                else:
                    im_tensor = img[None].to(device)

                x_tta, preds_seg, preds_rec, loss_dict = tta(im_tensor)

                # evaluate segmentation
                # Make sure it is the right shape
                # bs x it x c x h x w
                iou_losses = iou_loss(preds_seg.squeeze(), gt.squeeze().repeat(len(preds_seg), 1, 1, 1), thresh, apply_sigmoid=False)
                iou_losses = iou_losses.squeeze()
                print(iou_losses * 100)

                # plot results
                # plot_results_image(name, img, gt.squeeze(), preds_seg, preds_rec, iou_losses, loss_dict,
                #                    save_name=idx, folder=f'{corr}/{sev}/')
                print()

            #     so the plots don't consume too much memory when running on the server
            plt.close('all')




if __name__ == '__main__':
    args = get_segmentation_args(inference=True).parse_args()
    test_tta(args)
