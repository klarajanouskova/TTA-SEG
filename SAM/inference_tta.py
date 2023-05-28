import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse
from torchvision.transforms.v2.functional import to_pil_image


import sys
sys.path.append('..')
sys.path.append('modeling')

from distortion import distortions
from cityscapes_ext import PointCityscapes, PointCityscapesRain, PointCityscapesFog, cityscapes_root
from sam_tta import TestTimeAdaptor




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_sam_tta():
    parser = argparse.ArgumentParser('Segmentation-finetune', add_help=False)

    parser.add_argument('--tta_n_ims', default=1, type=int, help="number of images to use in TTA for optimization")
    parser.add_argument('--tta_ref_post', default=0, type=int, help="refinement post=processing")
    parser.add_argument('--tta_iter_num', default=10, type=int, help='number of iterations for SSL optimization')
    parser.add_argument('--tta_lr', default=1e-3, type=float, help='learning rate for SSL optimization')
    parser.add_argument('--model_path', default='sam_vit_b.pth', type=str)
    parser.add_argument('--loss_model', default="MaskLossNet", type=str, help='model class', choices=["MaskLossNet", "MaskLossUnet"])
    # # l1 with distortions l2 was in eval for this run()
    parser.add_argument('-tta_iou_model_run', default="sam_sam_deep_loss_adv_method_qual_conf80_min_sev0_segloss_IoU_trainloss_0.001_l1", type=str, help='weight decay for SSL optimization')
    parser.add_argument('-tta_ref_model_run', default="sam_sam_deep_loss_adv_method_ref_conf90_min_sev0_segloss_IoU_trainloss_0.001", type=str, help='weight decay for SSL optimization')
    parser.add_argument('--tta_optim', default="sgd", type=str, choices=["adam", "sgd"])
    parser.add_argument('--output_dir', default='/datagrid/TextSpotter/klara/TTA/ckpts',
                        help='path where to save, empty for no saving')
    return parser


def iou_loss(pred, mask, threshold=None, reduction='none', apply_sigmoid=True):
    assert reduction in ['mean', 'none']
    # because we are training with BCEwithLogitsLoss so sigmoid is not applied to the output
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    if threshold:
        pred = (pred > threshold).float()
    inter = (pred*mask).sum(dim=(2, 3))
    union = (pred+mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1)/(union-inter+1)
    if reduction == 'mean':
        return iou.mean()
    else:
        return iou


def sweep_small_eval_distorted(args):
    # TODO figure out why jpeg breakes with TypeError: unsupported operand type(s) for -: 'Image' and 'int', possibly update pillow, run for jpeg
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur', 'gaussian_blur', 'brightness', 'contrast', 'none']
    n_iter = args.tta_iter_num

    args.tta_iter_num = 4
    args.tta_lr = 1e-5
    # tta = TestTimeAdaptor(args=args, tta_method='ref')
    # tta_eval_cityscapes_distorted(args, tta=tta, samples=5, distortion_keys=distortion_keys, severity=3)

    args.tta_iter_num = n_iter

    # severities = [5, 3, 1]
    severities = [3, 2, 1]
    # severities = [1, 5]
    tent_lrs = [1e-2, 5e-3, 1e-3]
    sam_iou_lrs = [5e-4, 1e-4, 5e-5]
    # ref_lrs = [5e-6, 1e-5, 5e-5, 1e-4]

    #  test mask refinement post-processing, no real tta
    # for severity in severities:
    #     # for mask_ratio in [0.5, 0.6, 0.7, 0.75, 0.8]:
    #     for lr in tent_lrs:
    #         # args.mask_ratio = mask_ratio
    #         args.tta_lr = lr
    #         tta = TestTimeAdaptor(args=args, tta_method='tent')
    #         tta_eval_cityscapes_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)

    # test sam iou lr
    # for severity in severities:
    #     for lr in sam_iou_lrs:
    #         args.tta_lr = lr
    #         tta = TestTimeAdaptor(args=args, tta_method='sam-iou')
    #         tta_eval_cityscapes_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)

    for severity in severities:
        for lr in tent_lrs:
            args.tta_lr = lr
            tta = TestTimeAdaptor(args=args, tta_method='tent')
            tta_eval_cityscapes_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)

    # for severity in severities:
    #     for lr in ref_lrs:
    #         args.tta_lr = lr
    #         tta = TestTimeAdaptor(args=args, tta_method='ref')
    #         tta_eval_cityscapes_distorted(args, tta=tta, samples=20, distortion_keys=distortion_keys, severity=severity)

def benchmark_cityscapes():
    args = get_args_sam_tta().parse_args()
    # dataset = PointCityscapes(cityscapes_root, split='val', mode='fine', point_type='single',
    #                           target_type='instance')
    # # save_folder = 'base'
    # tta_eval_cityscapes_clean(dataset, args, save_folder)

    # dataset = PointCityscapesRain(cityscapes_root, split='val', mode='fine', point_type='single',
    #                               target_type='instance')
    # save_folder = 'rain'
    # tta_eval_cityscapes_clean(dataset, args, save_folder)

    dataset = PointCityscapesFog(cityscapes_root, split='val', mode='fine', point_type='single',
                                 target_type='instance')
    save_folder = 'fog'
    tta_eval_cityscapes_clean(dataset, args, save_folder)


def tta_eval_cityscapes_distorted(args, tta, samples=20, thresh=0.4, distortion_keys=['none'], severity=5):
    np.random.seed(0)
    dataset = PointCityscapes(cityscapes_root, split='val', mode='fine', point_type='single',
                              target_type='instance')

    folder = f'cityscapes_results/tta/corruptions'
    Path(folder).mkdir(parents=True, exist_ok=True)

    # no need for distribution shift here
    args.data_cls_sub = 'boat&cat&sheep&train'

    seg_iou_losses, deep_tta_losses = [], []

    for distortion_name in distortion_keys:
        corrupt_fun = distortions[distortion_name] if distortion_name != 'none' else None

        dist_seg_iou_losses, dist_deep_tta_losses = [], []
        # if n_samples are changed, we need to check the subdataset is good (no problematic gt)!
        for i in range(samples):
            image, gt_mask, points, labels = dataset[i]

            if len(points) == 0:
                continue

            # convert im to pil and apply corruption
            if corrupt_fun is not None:
                image = corrupt_fun(to_pil_image(image), severity=severity)
                image = image.round().astype(np.uint8)

            gt_mask = torch.tensor(gt_mask)
            # 'segmentation', 'segmentation_raw', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'

            xs_tta, tta_preds_seg, tta_loss_dict = tta(image, points, labels)

            ious = []
            # TODO we need to get the points that were actually predicted... get rid of all NMS for the sake of TTA?
            for pred, im_pts in zip(tta_preds_seg, points[:len(tta_preds_seg)]):
                px, py = int(im_pts[0][0] * pred.shape[2]), int(im_pts[0][1] * pred.shape[1])
                instance_gt = (gt_mask == gt_mask[py, px]).float()
                losses = iou_loss(pred[:, None], instance_gt.repeat(pred.shape[0], 1, 1, 1),
                                  reduction='none', apply_sigmoid=False, threshold=thresh)
                ious.append(losses.squeeze(-1))
            ious = torch.stack(ious, dim=0)
            print(ious)
            tmp_losses = []
            #     TODO fix this
            for k, v in tta_loss_dict.items():
                #  iter x nim, add extra dimension if necessary (nim=1)
                tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
            # loss x iter x nim -> nim x iter
            dist_deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)
            dist_seg_iou_losses.extend(ious)

        # nim x iter
        deep_tta_losses.append(dist_deep_tta_losses)
        seg_iou_losses.append(torch.stack(dist_seg_iou_losses))

    deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmena
    seg_iou_losses = np.array(torch.stack(seg_iou_losses)) * 100

    #   save numpy array with results
    save_name = f'{tta.save_name}_sev_{severity}'
    print(save_name)
    # array should be kind x im x iter - add extra 0 dim for kind
    np.save(f'{folder}/{save_name}.npy', [deep_tta_losses, seg_iou_losses])


def tta_eval_cityscapes_clean(dataset, args, save_folder, thresh=0.4,  n_samples=500):

    folder = f'cityscapes_results/tta/{save_folder}'
    Path(folder).mkdir(parents=True, exist_ok=True)

    # methods = ['ref', 'tent', 'sam-iou']
    methods = ['ref']
    lrs = {'tent': [5e-3], 'sam-iou': [5e-4], 'ref': [1e-4]}

    for method in methods:
        for lr in lrs[method]:
            args.tta_lr = lr
            tta = TestTimeAdaptor(args=args, tta_method=method, weights=[1.])

            deep_tta_losses, seg_iou_losses = [], []
            for i in range(n_samples):
                image, gt_mask, points, labels = dataset[i]
                if len(points) == 0:
                    continue
                gt_mask = torch.tensor(gt_mask)
                # 'segmentation', 'segmentation_raw', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'

                xs_tta, tta_preds_seg, tta_loss_dict = tta(image, points, labels)

                ious = []
                # TODO we need to get the points that were actually predicted... get rid of all NMS for the sake of TTA?
                for pred, im_pts in zip(tta_preds_seg, points):
                    px, py = int(im_pts[0][0] * pred.shape[2]), int(im_pts[0][1] * pred.shape[1])
                    instance_gt = (gt_mask == gt_mask[py, px]).float()
                    losses = iou_loss(pred[:, None], instance_gt.repeat(pred.shape[0], 1, 1, 1),
                                      reduction='none', apply_sigmoid=False, threshold=thresh)
                    ious.append(losses.squeeze(-1))
                ious = torch.stack(ious, dim=0)
                tmp_losses = []
                #     TODO fix this
                for k, v in tta_loss_dict.items():
                    #  iter x nim, add extra dimension if necessary (nim=1)
                    tmp_losses.append(v if len(v[0].shape) > 0 else [val[None] for val in v])
                # loss x iter x nim -> nim x iter
                deep_tta_losses.extend(np.array(tmp_losses).sum(0).T)
                seg_iou_losses.extend(ious)


            deep_tta_losses = np.array(deep_tta_losses).astype(float)  # make sure None becomes nan so that we can use nanmena
            seg_iou_losses = np.array(torch.stack(seg_iou_losses)) * 100

            #   save numpy array with results
            save_name = f'{tta.save_name}'
            print(save_name)
            # array should be kind x im x iter - add extra 0 dim for kind
            np.save(f'{folder}/{save_name}.npy', [deep_tta_losses[None], seg_iou_losses[None]])

if __name__ == '__main__':
    args = get_args_sam_tta().parse_args()
    dataset = PointCityscapes(cityscapes_root, split='val', mode='fine', point_type='single',
                              target_type='instance')
    save_folder = 'base'
    # tta_eval_cityscapes_clean(dataset, args, save_folder, n_samples=500)
    sweep_small_eval_distorted(args)
    # benchmark_cityscapes()