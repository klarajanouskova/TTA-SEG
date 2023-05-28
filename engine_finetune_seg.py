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

import util.misc as misc
import util.lr_sched as lr_sched
import wandb

from eval import Loss
from pascal_eval import test_pascal_subsets


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None, use_lr_sched=True,
                    args=None, pre='train'):
    model.train(True)
    loss_fun = Loss(args.loss, reduction='mean')

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('train/rec_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('train/seg_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    for loss in loss_fun.losses:
        metric_logger.add_meter(f'train/{loss}', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        im_samples = batch[0]
        masks = batch[1]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args, use_lr_sched)

        im_samples = im_samples.to(device, non_blocking=True)
        masks = masks.to(device)

        with torch.cuda.amp.autocast():
            loss_rec, _, _ = model.module.forward_rec(im_samples, mask_ratio=args.mask_ratio)
            preds_seg = model.module.forward_seg(im_samples)
            # preds_seg = torch.einsum('nchw->nhwc', preds_seg)
            # preds_seg = model.module.unpatchify(preds_seg)
            loss_seg, loss_vals_dict = loss_fun(preds_seg, masks)

        loss_total = args.rec_weight * loss_rec + args.seg_weight * loss_seg.mean()

        loss_value = loss_total.item()

        if not math.isfinite(loss_value):
            print("loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_total /= accum_iter
        loss_scaler(loss_total, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(**{f'train/loss': loss_total.item()})
        metric_logger.update(**{f'train/rec_loss': loss_rec.item()})
        metric_logger.update(**{f'train/seg_loss': loss_seg.mean().item()})

        for loss_val, loss_name in zip(loss_vals_dict.values(), loss_vals_dict.keys()):
            metric_logger.update(**{f'train/{loss_name}': loss_val.mean()})

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduction = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ we use epoch_1000x as the x-axis in tensorboard.
            this calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduction, epoch_1000x)
            log_writer.add_scalar('train/loss_rec', loss_rec, epoch_1000x)
            for loss_val, loss_name in zip(loss_vals_dict.values(), loss_vals_dict.keys()):
                log_writer.add_scalar('train/{}'.format(loss_name), loss_val, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, log_ims=False, pre='val', first_val=False):
    """
        Evaluation pipeline - only evaluates loss, the same as during training
    """
    # switch to evaluation mode
    model.eval()

    loss_fun = Loss(args.loss, reduction='mean')

    metric_logger = misc.MetricLogger(delimiter="  ")
    for loss in loss_fun.losses:
        metric_logger.add_meter(f'{pre}/{loss}', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter(f'{pre}/rec_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter(f'{pre}/seg_loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    result_dict = test_pascal_subsets(model.module, args, subsets=['A', 'B'], pre=pre, save_name=args.exp_name)

    #  go through keys in result dict and add them to metric logger
    for key, val in result_dict.items():
        metric_logger.add_meter(f'{key}', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
        metric_logger.update(**{f'{key}': val})

    b_idx = 0
    im_dict = {}

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, pre):
            images = batch[0]
            masks = batch[1]

            images = images.to(device, non_blocking=True)
            masks = masks.to(device)

            # compute output
            with torch.cuda.amp.autocast():
                loss_rec, _, _ = model.module.forward_rec(images, mask_ratio=args.mask_ratio)
                preds_seg = model.module.forward_seg(images)
                loss_seg, loss_vals_dict = loss_fun(preds_seg, masks)
            loss_total = args.rec_weight * loss_rec + args.seg_weight * loss_seg.mean()

            torch.cuda.synchronize()

            metric_logger.update(**{f'{pre}/loss': loss_total.item()})
            metric_logger.update(**{f'{pre}/rec_loss': loss_rec.item()})
            metric_logger.update(**{f'{pre}/seg_loss': loss_seg.mean().item()})
            for loss_val, loss_name in zip(loss_vals_dict.values(), loss_vals_dict.keys()):
                metric_logger.update(**{f'{pre}/{loss_name}': loss_val.mean()})

            if b_idx == 0:
                n = images.size(0)
                n_vis = min(n, 3)
                ims_vis, segs_vis, masks_vis = [], [], []
                for i in range(n_vis):
                    # ims_vis.append()
                    # segs_vis.append()
                    # masks_vis.append(wandb.Image(
                    #     torch.einsum('chw->hwc', masks[i]).detach().cpu().numpy(), caption=f'gt {i}'))
                    if first_val:
                        # only log input image once
                        im_dict[f"{pre}_ims/in{i}"] = [wandb.Image(
                            torch.einsum('chw->hwc', images[i]).detach().cpu().numpy())]
                    im_dict[f"{pre}_ims/out{i}"] = [wandb.Image(
                        torch.einsum('chw->hwc', preds_seg[i]).detach().cpu().numpy())]
                # im_dict[f"{pre}seg GT"] = masks_vis
            b_idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* Loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.meters[f'{pre}/loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()} | im_dict









