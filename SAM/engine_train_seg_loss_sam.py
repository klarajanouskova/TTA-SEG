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
import matplotlib.pyplot as plt

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


def get_predicted_loss(model_loss, loss_fun, masks, dist_masks, args, thresh=0.4):
    dist_masks = torch.stack(dist_masks)[:, None]
    masks = torch.stack(masks)[:, None]
    if args.binarize_pred_gt:
        masks = (masks > thresh).float()
    seg_loss_gt, loss_dict = loss_fun(dist_masks, masks )

    if args.learn_method == 'qual':
        predicted_loss = model_loss(dist_masks)
        if args.learn_loss == 'l1':
            learn_loss = F.l1_loss(predicted_loss, seg_loss_gt, reduction='none')
        elif args.learn_loss == 'l2':
            learn_loss = F.mse_loss(predicted_loss, seg_loss_gt, reduction='none')
        else:
            raise ValueError('Unknown loss type')

    elif args.learn_method == 'ref':
        pred_masks_ref = model_loss(dist_masks)
        learn_loss = loss_fun(pred_masks_ref, masks)[0]

    return learn_loss, seg_loss_gt


def get_adversarial_image(model_seg, img, pts, labels):
    pts = torch.tensor([pts], dtype=torch.float32, device=model_seg.predictor.device)
    labels = torch.tensor([labels], dtype=torch.int64, device=model_seg.predictor.device)

    # scale pts by image size
    pts = pts * torch.tensor([img.shape[1], img.shape[0]], dtype=torch.float32,
                             device=model_seg.predictor.device)
    # sample parameters
    gt_kind = np.random.choice(['invert', 'random'])
    # make porbability of more expensive operations lower
    ps = 1 / (np.arange(11) + 2)
    ps = ps / ps.sum()
    if gt_kind == 'invert':
        iters = np.random.choice(np.arange(11), p=ps)
        lr = 0.0001
    else:
        iters = np.random.choice(np.arange(11), p=ps)
        # so that we  do less iterations
        lr = 0.0001
    dist = model_seg.pgd_attack(img,  pts, labels, iters=iters, lr=lr, debug=False, gt=gt_kind)
    return dist


def train_one_epoch(model_loss: torch.nn.Module, model_seg: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    log_writer=None, use_lr_sched=True,
                    args=None, pre='train'):

    model_loss.train(True)
    loss_fun = Loss('IoU', reduction='none', apply_sigmoid=False)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train/loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter(f'train/seg_err', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    filtered_masks, filtered_dist_masks = [], []
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image, gt_mask, points, labels = [e[0] for e in batch]
        # image to pil
        image = image.cpu().numpy()
        # points, labels to numpy
        points = points.cpu().numpy()
        labels = labels.cpu().numpy()
        
        if len(points) == 0:
            continue

        with torch.no_grad():
            sam_out = model_seg.generate(image, points, labels)
        # TODO aggregate to single tensor, run rec only once
        masks, scores = [], []

        for sam_pred in sam_out:
            pred_mask = sam_pred['segmentation_raw']
            score = sam_pred['predicted_iou']
            masks.append(pred_mask)
            scores.append(score)

        # only keep confident masks
        keep = [score > args.sam_conf for score in scores]
        masks = [mask for mask, k in zip(masks, keep) if k]
        scores = [score for score, k in zip(scores, keep) if k]
        points = [point for point, k in zip(points, keep) if k]
        labels = [label for label, k in zip(labels, keep) if k]
        top_idxs = torch.argsort(torch.tensor(scores), 0, descending=True)

        filtered_masks += [masks[i] for i in top_idxs]
        dist_imgs = [get_adversarial_image(model_seg, image, points[i], labels[i]) for i in top_idxs]

        dist_masks = []
        for idx, dist_img in enumerate(dist_imgs):
            with torch.no_grad():
                sam_out = model_seg.generate(dist_img, points[top_idxs[idx]][None], labels[top_idxs[idx]][None])
                dist_masks.append(sam_out[0]['segmentation_raw'])
        filtered_dist_masks += dist_masks
        dist_masks, dist_imgs = None, None

        # free up memory
        masks, points, labels = None, None, None

        if len(filtered_masks) > args.batch_size:

            learnt_loss, seg_err = get_predicted_loss(model_loss, loss_fun, filtered_masks[:args.batch_size], filtered_dist_masks[:args.batch_size], args)

            # do not throw away the rest
            filtered_masks = filtered_masks[args.batch_size:]

            filtered_dist_masks = filtered_dist_masks[args.batch_size:]

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

    n, sum, sum_sq = 0, 0, 0

    filtered_masks, filtered_dist_masks = [], []
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, 1, pre)):
        image, gt_mask, points, labels = [e[0] for e in batch]
        # image to pil
        image = image.cpu().numpy()
        # points, labels to numpy
        points = points.cpu().numpy()
        labels = labels.cpu().numpy()

        if len(points) == 0:
            continue

        with torch.no_grad():
            sam_out = model_seg.generate(image, points, labels)
        # TODO aggregate to single tensor, run rec only once
        masks, scores = [], []
        for sam_pred in sam_out:
            pred_mask = sam_pred['segmentation_raw']
            score = sam_pred['predicted_iou']
            masks.append(pred_mask)
            scores.append(score)

        # only keep confident masks
        keep = [score > args.sam_conf for score in scores]
        masks = [mask for mask, k in zip(masks, keep) if k]
        scores = [score for score, k in zip(scores, keep) if k]
        points = [point for point, k in zip(points, keep) if k]
        labels = [label for label, k in zip(labels, keep) if k]
        top_idxs = torch.argsort(torch.tensor(scores), 0, descending=True)

        filtered_masks += [masks[i] for i in top_idxs]
        dist_imgs = [get_adversarial_image(model_seg, image, points[i], labels[i]) for i in top_idxs]

        dist_masks = []
        for idx, dist_img in enumerate(dist_imgs):
            with torch.no_grad():
                sam_out = model_seg.generate(dist_img, points[top_idxs[idx]][None], labels[top_idxs[idx]][None])
                dist_masks.append(sam_out[0]['segmentation_raw'])
        filtered_dist_masks += dist_masks
        dist_masks = None
        # free up memory
        masks, points, labels = None, None, None

        if len(filtered_masks) > args.batch_size:
            learnt_loss, seg_err = get_predicted_loss(model_loss, loss_fun, filtered_masks[:args.batch_size], filtered_dist_masks[:args.batch_size], args)

            n += len(learnt_loss)
            sum += learnt_loss.sum().item()
            sum_sq += (learnt_loss ** 2).sum().item()

            loss_total = learnt_loss.mean()

            torch.cuda.synchronize()

            metric_logger.update(**{f'{pre}/loss': loss_total.item()})
            metric_logger.update(**{f'{pre}/seg_err': seg_err.mean().item()})

            if b_idx == 0 and log_ims and args.learn_method == 'ref':
                n = args.batch_size
                n_vis = min(n, 6)
                for i in range(n_vis):
                     # make preds_seg_dist[i] rgb so that we can concat it with im_samples[i] and masks[i]
                    pred_cc = torch.stack([filtered_dist_masks[i], filtered_dist_masks[i], filtered_dist_masks[i]]).detach().cpu()
                    pred_cc = torch.einsum('chw->hwc', pred_cc)
                     # same for masks[i]
                    mask_cc = torch.stack([filtered_masks[i], filtered_masks[i], filtered_masks[i]]).detach().cpu()
                    mask_cc = torch.einsum('chw->hwc', mask_cc)

                     # concat im, gt and pred horizontally into a single image
                    composed = (torch.cat([pred_cc * 255, mask_cc * 255], dim=0)).int()
                    im_dict[f"{pre}_ims/composed{i}"] = [wandb.Image(composed.numpy())]

            # do not throw away the rest
            filtered_masks = filtered_masks[args.batch_size:]

            filtered_dist_masks = filtered_dist_masks[args.batch_size:]

            b_idx += 1
        # at least 250 validation images
        if b_idx >= (250 // args.batch_size):
            break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.meters[f'{pre}/loss']))

    # compute the mean and std of the loss
    mean = sum / n
    std = np.sqrt(sum_sq / n - mean ** 2)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} | im_dict | {'pred_loss_mean': mean, 'pred_loss_std': std}









