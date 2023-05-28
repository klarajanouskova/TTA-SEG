# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision.transforms.v2.functional import to_pil_image, to_tensor


import util.misc as misc
import util.lr_sched as lr_sched
import wandb
from distortion import get_random_corruption_fun, pgd

from eval import Loss
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO make normalization general so that it can be used for batches as well (maybe it already is, check)
def denormalize_tensor(img):
    """
    Transform image-net normalized image to original [0, 1] range
    """
    return img * torch.tensor(IMAGENET_DEFAULT_STD, device=img.device)[:, None, None] + \
           torch.tensor(IMAGENET_DEFAULT_MEAN, device=img.device)[:, None,
                                                                     None]

def normalize_tensor(img):
    """
    Transform image-net normalized image to original [0, 1] range
    """
    return (img - torch.tensor(IMAGENET_DEFAULT_MEAN, device=img.device)[:, None, None]) / \
           torch.tensor(IMAGENET_DEFAULT_STD, device=img.device)[:, None, None]


def get_predicted_loss(model_seg, model_loss, loss_fun, im_samples, dist_im_samples, masks, args, thresh=0.4):
    with torch.no_grad():
        preds_seg_dist = model_seg.forward_seg(dist_im_samples, inference=True)
        # inference on clean images is considered GT
        # we may want to use this instead of GT masks
        # pred_seg_clean = model_seg.forward_seg(im_samples, inference=True)
        if args.loss_use_gt:
            seg_loss_gt, loss_dict = loss_fun(preds_seg_dist, masks)
        else:
            preds_seg_clean = model_seg.forward_seg(im_samples, inference=True)
            if args.binarize_pred_gt:
                preds_seg_clean = (preds_seg_clean > thresh).float()
            seg_loss_gt, loss_dict = loss_fun(preds_seg_dist, preds_seg_clean)

    if args.learn_method == 'qual':
        predicted_loss = model_loss(preds_seg_dist)
        if args.learn_loss == 'l1':
            learn_loss = F.l1_loss(predicted_loss, seg_loss_gt, reduction='none')
        elif args.learn_loss == 'l2':
            learn_loss = F.mse_loss(predicted_loss, seg_loss_gt, reduction='none')
        else:
            raise ValueError('Unknown loss type')

    elif args.learn_method == 'ref':
        pred_masks_ref = model_loss(preds_seg_dist)
        # [0] to get the aggregated loss
        if args.loss_use_gt:
            learn_loss = loss_fun(pred_masks_ref, masks)[0]
        else:
            learn_loss = loss_fun(pred_masks_ref, preds_seg_clean)[0]

    return learn_loss, seg_loss_gt, preds_seg_dist


def get_distorted_samples(args, model_seg, im_samples, severities):
    if args.domain_shift_method == 'corruption':
        # makes use of the known corruption methods
        # TODO parametrize this
        # apply random distortion to each image in batch, denormalize them before and then normalize again
        distorted = []
        # first undo normalization
        denorm_ims = im_samples * torch.tensor(IMAGENET_DEFAULT_STD, device=im_samples.device).view(3, 1, 1) + \
                     torch.tensor(IMAGENET_DEFAULT_MEAN, device=im_samples.device).view(3, 1, 1)
        for im in denorm_ims:
            dist = get_random_corruption_fun(severities)(to_pil_image(im))
            #   renormalize, go back to tensor
            dist = (dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)
            distorted.append(to_tensor(dist).float())
        dist_im_samples = torch.stack(distorted, dim=0) / 255
    elif args.domain_shift_method == 'adversarial':
        distorted = []
        for im in im_samples:
            # sample parameters
            gt_kind = np.random.choice(['invert', 'random'])
            # make porbability of more expensive operations lower
            ps = 1 / (np.arange(11) + 2)
            ps = ps / ps.sum()
            if gt_kind == 'invert':
                iters = np.random.choice(np.arange(11), p=ps)
                lr = 0.001
            else:
                iters = np.random.choice(np.arange(11), p=ps)
                # so that we  do less iterations
                lr = 0.005
            dist = pgd(model_seg, im[None], iters=iters, gt=gt_kind, lr=lr, norm_fun=normalize_tensor, inv_norm_fun=denormalize_tensor)
            distorted.append(dist)
        dist_im_samples = torch.cat(distorted, dim=0)
    else:
        raise ValueError('Unknown domain shift method')

    return dist_im_samples


def train_one_epoch(model_loss: torch.nn.Module, model_seg: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None, use_lr_sched=True,
                    args=None, pre='train'):

    model_loss.train(True)
    loss_fun = Loss(args.loss, reduction='none', apply_sigmoid=False)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train/loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter(f'train/seg_err', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    severities = np.arange(args.min_dist, args.max_dist + 1)

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        im_samples = batch[0]
        masks = batch[1]

        im_samples = im_samples.to(device, non_blocking=True)
        dist_im_samples = get_distorted_samples(args, model_seg, im_samples, severities)
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args, use_lr_sched)

        dist_im_samples = dist_im_samples.to(device, non_blocking=True)
        masks = masks.to(device)

        learnt_loss, seg_err, preds_seg_dist = get_predicted_loss(model_seg, model_loss, loss_fun, im_samples, dist_im_samples, masks, args)

        loss_total = learnt_loss.mean()

        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total /= accum_iter

        loss_total.backward()
        optimizer.step()

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{f'train/loss': loss_total.item()})
        metric_logger.update(**{f'train/seg_err': seg_err.mean().item()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduction = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ we use epoch_1000x as the x-axis in tensorboard.
            this calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduction, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate(data_loader, model_loss, model_seg, device, args, log_ims=False, pre='val', first_val=False):
    """
        Evaluation pipeline - only evaluates loss, the same as during training
    """
    # switch to evaluation mode
    model_loss.eval()

    loss_fun = Loss('IoU', reduction='none', apply_sigmoid=False)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(f'{pre}/loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter(f'{pre}/seg_err', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    # fix np (corruption) and torch (adversarial) seed because of the distortions
    np.random.seed(0)
    torch.manual_seed(0)

    b_idx = 0
    im_dict = {}
    severities = np.arange(args.min_dist, args.max_dist + 1)

    n, sum, sum_sq = 0, 0, 0

    for batch in metric_logger.log_every(data_loader, 10, pre):
        im_samples = batch[0]
        masks = batch[1]

        im_samples = im_samples.to(device, non_blocking=True)
        masks = masks.to(device)

        dist_im_samples = get_distorted_samples(args, model_seg, im_samples, severities)

        learnt_loss, seg_err, preds_seg_dist = get_predicted_loss(model_seg, model_loss, loss_fun, im_samples, dist_im_samples, masks, args)

        n += im_samples.size(0)
        sum += learnt_loss.sum().item()
        sum_sq += (learnt_loss ** 2).sum().item()

        loss_total = learnt_loss.mean()

        torch.cuda.synchronize()

        metric_logger.update(**{f'{pre}/loss': loss_total.item()})
        metric_logger.update(**{f'{pre}/seg_err': seg_err.mean().item()})

        if b_idx == 0 and log_ims and args.learn_method == 'ref':
            dist_im_samples = dist_im_samples.detach().cpu()
            denorm_dist = dist_im_samples * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                     torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
            n = im_samples.size(0)
            n_vis = min(n, 6)
            ims_vis, segs_vis, masks_vis = [], [], []
            for i in range(n_vis):
                # if first_val:
                #     im_dict[f"{pre}_ims/in{i}"] = [wandb.Image(
                #         torch.einsum('chw->hwc', im_samples[i]).numpy())]
                # im_dict[f"{pre}_ims/out{i}"] = [wandb.Image(
                #     torch.einsum('chw->hwc', preds_seg_dist[i]).detach().cpu().numpy())]

                im_cc = denorm_dist[i]
                 # make preds_seg_dist[i] rgb so that we can concat it with im_samples[i] and masks[i]
                pred_cc = torch.vstack([preds_seg_dist[i], preds_seg_dist[i], preds_seg_dist[i]]).detach().cpu()
                 # same for masks[i]
                mask_cc = torch.vstack([masks[i], masks[i], masks[i]]).detach().cpu()

                 # concat im, gt and pred horizontally into a single image
                composed = (torch.cat([im_cc, pred_cc, mask_cc], dim=2) * 255).int()
                im_dict[f"{pre}_ims/composed{i}"] = [wandb.Image(
                    torch.einsum('chw->hwc', composed).numpy())]

        b_idx += 1
        # there is too many validation images in pascal that jsut slow down training...
        if b_idx == 40:
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.meters[f'{pre}/loss']))

    # compute the mean and std of the loss
    mean = sum / n
    std = np.sqrt(sum_sq / n - mean ** 2)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} | im_dict | {'pred_loss_mean': mean, 'pred_loss_std': std}









