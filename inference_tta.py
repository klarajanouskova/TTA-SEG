"""
Inference time self-supervised learning
"""

import sys
import os

import torch
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2.functional import to_pil_image, to_tensor

import numpy as np
import matplotlib.pyplot as plt
# prevent matpltolib form using scientific notation
plt.rcParams['axes.formatter.useoffset'] = False


from tqdm import tqdm


from util.datasets_seg import get_pascal

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from eval import iou_loss
from util.voc_dataset_seg import CA_CLEAN_IDXS_VAL_A

from distortion import distortions
from tta import TestTimeAdaptor

from arg_composition import get_segmentation_args

sys.path.append('..')

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


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


def eval_distorted(args, tta, thresh=0.4, samples=20, distortion_keys=['none'], severity=5):
    # check that samples is a multiple of tta_n_ims
    assert samples % args.tta_n_ims == 0
    # check that there are at least tta_n_ims samples
    assert samples >= args.tta_n_ims, f'Need at least {args.tta_n_ims} samples for TTA, but only {samples} were requested'

    np.random.seed(0)
    folder = f'pascal_tta_clean_{samples}_distorted'
    os.makedirs(folder, exist_ok=True)

    # no need for distribution shift here
    args.data_cls_sub = 'boat&cat&sheep&train'
    dataset = get_pascal(args, split='val')

    seg_iou_losses, deep_tta_losses = [], []

    for distortion_name in distortion_keys:
        corrupt_fun = distortions[distortion_name] if distortion_name != 'none' else None

        dist_seg_iou_losses, dist_deep_tta_losses = [], []
        im_tensors, gts = [], []
        # if n_samples are changed, we need to check the subdataset is good (no problematic gt)!
        for c, idx in tqdm(enumerate(CA_CLEAN_IDXS_VAL_A[:samples])):
            img, gt, cls, name = dataset[idx]

            gt = (gt > 0).int()

            # distort and add batch dimension
            if corrupt_fun is not None:
                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
                dist = corrupt_fun(to_pil_image(denorm_im), severity=severity) / 255
                #   renormalize, go back to tensor
                img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                im_tensor = img[None].float().to(device)
            else:
                im_tensor = img[None].to(device)

            im_tensors.append(im_tensor)
            gts.append(gt)

            if (c + 1) % args.tta_n_ims == 0:
                im_tensors = torch.vstack(im_tensors)
                #  bs x it x c x h x w
                xs_tta, preds_seg, preds_rec, loss_dict = tta(im_tensors)

                # evaluate segmentation
                # Make sure it is the right shape
                for im_preds_seg, gt in zip(preds_seg, gts):
                    im_seg_iou_losses = iou_loss(im_preds_seg, gt.repeat(len(im_preds_seg), 1, 1, 1), thresh, apply_sigmoid=False)
                    # its x nim
                    im_seg_iou_losses = im_seg_iou_losses.squeeze(-1).numpy()
                    print(im_seg_iou_losses * 100)

                    dist_seg_iou_losses.append(im_seg_iou_losses)
                    # sum losses from loss dict and add to dist_deep_tta_losses
                tmp_losses = []
                #     TODO fix this
                for k, v in loss_dict.items():
                    #  iter x nim, add extra dimension if necessary (nim=1)
                    tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
                # loss x iter x nim -> nim x iter
                dist_deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)

                #     reset
                im_tensors, gts = [], []
                if idx < 10:
                    pass
                    # TODO loop over images
                    # im_save_name = f'{distortion_name}{severity}_{idx}_{name}'
                    # plot_results_image(name, img, gt.squeeze(), im_preds_seg, im_preds_rec, im_rec_mses, im_seg_deep_losses, im_seg_iou_losses, save_name=im_save_name)

        # nim x iter
        deep_tta_losses.append(dist_deep_tta_losses)
        seg_iou_losses.append(dist_seg_iou_losses)

    # convert all results to np arrays
    deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmena
    seg_iou_losses = np.array(seg_iou_losses) * 100

    print('-' * 100)
    print(f'Average iou losses before TTA: {np.nanmean(seg_iou_losses[:, :, 0])};'
          f' after TTA: {np.nanmean(seg_iou_losses[:, :, -1])}')
    # kind x im x iter
    # print(seg_iou_losses[:, :, 0])
    # print(seg_iou_losses[:, :, -1])
    print('-' * 100)


    #   save numpy array with results
    save_name = f'{tta.save_name}_sev_{severity}'
    print(save_name)
    # array should be kind x im x iter
    np.save(f'{folder}/{save_name}.npy', [deep_tta_losses, seg_iou_losses])


def sweep_small_eval_distorted(args):
    # TODO figure out why jpeg breakes with TypeError: unsupported operand type(s) for -: 'Image' and 'int', possibly update pillow, run for jpeg
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur', 'gaussian_blur', 'brightness', 'contrast', 'none']
    n_iter = args.tta_iter_num

    # baseline - no TTA
    args.tta_iter_num = 3
    # tta_iou_opt_sgd_lr_0.05_freeze_rec_1_its_10_gradclip_0.5_nims_1_sev_5
    # args.tta_grad_clip = 0.5
    # args.tta_n_ims = 1
    # tta = TestTimeAdaptor(args=args, tta_method='rec')
    # small_eval_distorted(args, tta=tta, samples=20, distortion_keys=['none'])
    # tta = TestTimeAdaptor(args=args, tta_method='tent')
    # eval_distorted(args, tta=tta, samples=2, distortion_keys=['none', 'contrast'])

    args.tta_iter_num = n_iter

    severities = [5, 3, 1]
    tent_lrs = [1e-3]
    adv_lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    iou_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    ref_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # rec_lrs = [7e-2]
    grad_clips = [-1]

    # non-frozen encoder
    # args.tta_freeze_rec = 0
    # for severity in severities:
    #     for lr in rec_lrs:
    #         args.tta_lr = lr
    #         tta = TestTimeAdaptor(args=args, tta_method='rec')
    #         eval_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)


    #  test mask refinement post-processing, no real tta
    args.tta_freeze_rec = 1
    for severity in severities:
        # for tent_lrs in [0.5, 0.6, 0.7, 0.75, 0.8]:
        for lr in tent_lrs:
            # args.mask_ratio = mask_ratio
            args.tta_lr = lr
            tta = TestTimeAdaptor(args=args, tta_method='tent')
            eval_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)


def sweep_method_combination(args):
    # TODO figure out why jpeg breakes with TypeError: unsupported operand type(s) for -: 'Image' and 'int', possibly update pillow, run for jpeg
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur',
                       'gaussian_blur', 'brightness', 'contrast', 'none']

    methods = ['rec', 'ref', 'iou']
    rec_lr = 5e-2
    ref_lr = 1e-3
    iou_lr = 5e-4
    rec_weight = 1
    # adjust weight so that we can use rec lr
    ref_weight = ref_lr / rec_lr
    iou_weight = iou_lr / rec_lr
    # ref_weight = 0
    n_iter = args.tta_iter_num

    # baseline - no TTA
    args.tta_iter_num = 10
    # args.tta_rec_batch_size = 3

    # tta = TestTimeAdaptor(args=args, tta_method='&'.join(methods), weights=[rec_weight, ref_weight, iou_weight])
    # eval_distorted(args, tta=tta, samples=5, distortion_keys=['spatter'])

    args.tta_iter_num = n_iter

    severities = [3, 1, 5]
    optim = 'sgd'
    args.tta_freeze_rec = 1
    for severity in severities:
        lr = rec_lr
        args.tta_optim = optim
        args.tta_lr = lr
        tta = TestTimeAdaptor(args=args, tta_method='&'.join(methods), weights=[rec_weight, ref_weight, iou_weight])
        eval_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)


def sweep_batch_eval_distorted(args):
    # TODO figure out why jpeg breakes with TypeError: unsupported operand type(s) for -: 'Image' and 'int', possibly update pillow, run for jpeg

    n_iter = args.tta_iter_num
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur',
                       'gaussian_blur', 'brightness', 'contrast', 'none']

    # baseline - no TTA
    args.tta_iter_num = 2
    # args.tta_rec_batch_size = 3
    args.tta_n_ims = 2

    #tta_iou_opt_sgd_lr_0.05_freeze_rec_1_its_10_gradclip_-1_nims_5_sev_3

    # tta = TestTimeAdaptor(args=args, tta_method='rec')
    # batch_eval_distorted(args, tta=tta, samples=8, distortion_keys=['none'])
    # tta = TestTimeAdaptor(args=args, tta_method='iou')
    # eval_distorted(args, tta=tta, samples=4, distortion_keys=['none', 'fog'], severity=3)
    # tta = TestTimeAdaptor(args=args, tta_method='tent&rec')
    # batch_eval_distorted(args, tta=tta, samples=6, distortion_keys=['none'], severity=3)

    args.tta_iter_num = n_iter

    batch_sizes = [1, 3, 5]
    # batch_sizes = [8]
    # severities = [5, 3, 1]
    severities = [3, 1, 5]
    optims = ['sgd']
    lrs = [5e-2, 1e-2]
    method_lrs = {'tent': 5e-3, 'rec': 5e-2, 'ref': 1e-3, 'iou': 5e-4}
    grad_clips = [-1]
    args.tta_freeze_rec = 1
    for severity in severities:
        for optim in optims:
            for method in ['iou', 'ref']:
                lr = method_lrs['iou']
                for grad_clip in grad_clips:
                    for batch_size in batch_sizes:
                        tta_rec_batch_size = 8 // batch_size
                        args.tta_rec_batch_size = tta_rec_batch_size
                        args.tta_n_ims = batch_size
                        # TODO set args batch size
                        args.tta_grad_clip = grad_clip
                        args.tta_optim = optim
                        args.tta_lr = lr
                        tta = TestTimeAdaptor(args=args, tta_method=method)
                        eval_distorted(args, tta=tta, samples=120, distortion_keys=distortion_keys, severity=severity)
                        # tta = TestTimeAdaptor(args=args, tta_method='iou')
                        # eval_distorted(args, tta=tta, samples=120, distortion_keys=distortion_keys, severity=severity)
                        # tta = TestTimeAdaptor(args=args, tta_method='ref')
                        # small_eval_distorted(args, tta=tta, samples=120, distortion_keys=distortion_keys, severity=severity)
                        # tta = TestTimeAdaptor(args=args, tta_method='tent')
                        # eval_distorted(args, tta=tta, samples=120, distortion_keys=distortion_keys, severity=severity)


def eval_seg(preds, gt, thresh=0.4):
    ious = []
    for pred in preds:
        pred = (pred > thresh).int()
        iou = (pred * gt).sum() / ((pred + gt) > 0).sum()
        ious.append(1 - iou)
    return ious


def im_to_vis(im):
    denorm_im = im * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                         torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    im = F.to_pil_image(denorm_im)
    return im


def denormalize(img):
    """
    Transform image-net normalized image to original [0, 1] range
    """
    return img * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]


if __name__ == '__main__':
    args = get_segmentation_args().parse_args()
    args.run_name = 'sweep_aspect2_pascal_A_SEG+REC_ps_16_p_ar'
    # sweep_small_eval_class(args)
    # sweep_batch_eval_distorted(args)
    # sweep_method_combination(args)
    sweep_small_eval_distorted(args)
    # main(args)

