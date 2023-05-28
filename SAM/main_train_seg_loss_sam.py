# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
# to enable imports from parent
import sys
sys.path.append('..')
sys.path.append('modeling')
from pathlib import Path
sys.path.append(str(Path('.').absolute().parent))

import datetime

import numpy as np
import os
import time
from pathlib import Path
import argparse

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import wandb

from cityscapes_ext import PointCityscapes, cityscapes_root
from engine_train_seg_loss_sam import train_one_epoch, evaluate

from models_loss import MaskLossUnet, MaskLossNet

from util import misc

from eval_segmentation import load_sam

local = not torch.cuda.is_available()

import torch.nn as nn


def get_args_sam_train_tta():
    parser = argparse.ArgumentParser('Segmentation-finetune', add_help=False)

    parser.add_argument('--exp_name', default='sam_surrogate', type=str, help="experiment name")
    parser.add_argument('--tta_n_ims', default=1, type=int, help="number of images to use in TTA for optimization")
    parser.add_argument('--tta_iter_num', default=10, type=int, help='number of iterations for SSL optimization')
    parser.add_argument('--tta_lr', default=1e-3, type=float, help='learning rate for SSL optimization')
    parser.add_argument('--model_path', default='sam_vit_b.pth', type=str)
    
    parser.add_argument('--learn_loss', default="l1", type=str, choices=["l1", "l2"])
    # quality or refinement estimation
    parser.add_argument('--learn_method', default="ref", type=str, choices=["ref", "qual"])
    parser.add_argument('--min_dist', default=0, type=int, help="distortion severity lower limit")
    parser.add_argument('--sam_conf', default=0.8, type=float, help="confidence threshold on predictions")
    parser.add_argument('--max_dist', default=5, type=int, help="distortion severity upper limit")
    parser.add_argument('--loss_use_gt', default=0, type=int, help="whether to compute loss wrt clean image mask, or the gt mask")
    parser.add_argument('--binarize_pred_gt', default=0, type=int, help="whether to binarize clean image mask before loss computation")
    parser.add_argument('--domain_shift_method', default='adversarial', type=str, help="which domain shift method to use", choices=["adversarial", "corruption"])
    
    parser.add_argument('--output_dir', default='/datagrid/TextSpotter/klara/TTA/ckpts',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint to continue training')
    parser.add_argument('--num_workers', default=0, type=int)

    # distribUuted training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--batch_size', default=2, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # optimization params
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                    help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')

    return parser


def get_model(args):
    input_size = (512, 1024)
    if args.learn_method == 'qual':
        return MaskLossNet(input_size)
    elif args.learn_method == 'ref':
       return MaskLossUnet()


def get_train_dataloader(dataset, args, global_rank, num_tasks):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        # bs collected from instances
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return data_loader_train


def get_test_dataloader(dataset, args):
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_val,
        # bs collected from instances
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader_test


def main(args, init=True):
    if not local:
        print(torch.cuda.memory_allocated())
    if local:
        args.num_workers = 0

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # for local debugging
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    dataset = PointCityscapes(cityscapes_root, split='train', mode='fine', point_type='single',
                              target_type='instance')
    # split dataset into train and val
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [len(dataset) - 500, 500])
    data_loader_train, data_loader_val = get_train_dataloader(dataset_train, args, global_rank, num_tasks), get_test_dataloader(dataset_val, args)

    model_loss = get_model(args)

    if global_rank == 0:
        # has to be done before the summarywriter is created!
        wandb.init(project="TTA-finetune", entity="klara", name=args.exp_name, reinit=True)
        wandb.config = vars(args)
        if args.output_dir is not None:
            args.log_dir = os.path.join(args.output_dir, args.exp_name, 'logs')
            os.makedirs(args.log_dir, exist_ok=True)
            if init:
                wandb.tensorboard.patch(root_logdir=args.log_dir, pytorch=True, save=False)
            log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # load pretrained segmentation model
    model_seg = load_sam(args.model_path)

    model_loss.to(device)
    model_seg.eval()

    # freeze all seg  model params
    for param in model_seg.predictor.model.parameters():
        param.requires_grad = False

    model_without_ddp = model_loss
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model_loss = torch.nn.parallel.DistributedDataParallel(model_loss, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model_loss.module
    elif not local:
        model_loss = torch.nn.DataParallel(model_loss)
        model_without_ddp = model_loss.module

    if global_rank == 0:
        wandb.watch(model_without_ddp)

    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, betas=(0.9, 0.95))

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss_train, min_loss_val = np.inf, np.inf

    if data_loader_val:
        # evaluate before we start training
        val_stats = evaluate(data_loader_val, model_loss, model_seg, device, args, log_ims=True, pre='val', first_val=True)
        print(f"Validation loss before training: {val_stats['val/loss']:.1f}")
        min_loss_val = min(min_loss_val, val_stats["val/loss"])
        misc.save_model(
            args=args, model=model_loss, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=0, best=True)

        if global_rank == 0:
            wandb.log({"val/min_loss": min_loss_val})
            wandb.log(val_stats)
            wandb.run.summary["val/min_loss"] = min_loss_val

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model_loss, model_seg, data_loader_train,
            optimizer, device, epoch,
            log_writer=log_writer,
            use_lr_sched=False,
            args=args
        )

        min_loss_train = min(min_loss_train, train_stats["train/loss"])

        print(f'Min loss train: {min_loss_train:.2f}%')

        if data_loader_val:
            val_stats = evaluate(data_loader_val, model_loss, model_seg, device, args, log_ims=True, pre='val')
            print(f"Loss of the network on the validation images: {val_stats['val/loss']:.1f}%")

            if min_loss_val > val_stats["val/loss"]:
                min_loss_val = val_stats["val/loss"]
                misc.save_model(
                    args=args, model=model_loss, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, best=True)
            val_stats["val/min_loss"] = min_loss_val

            print(f'Min loss val: {min_loss_val:.2f}%')


        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()

        if global_rank == 0:
            if data_loader_val:
                wandb.log(val_stats)
                # wandb.log({"val/loss": val_stats["loss"]})
                wandb.run.summary["val/min_loss"] = min_loss_val
            wandb.log({"train/min_loss": min_loss_train})
            wandb.run.summary["train/min_loss"] = min_loss_train

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    misc.save_model(args=args, model=model_loss, model_without_ddp=model_without_ddp, optimizer=optimizer, epoch=epoch, last=True)

    wandb.finish()


if __name__ == '__main__':
    # Launch: python -m torch.distributed.launch --nproc_per_node=4 main_finetune_self.py --world_size 4 --model mae_vit_base_patch16
    args = get_args_sam_train_tta().parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    c = 0
    start_c = 0
    args.exp_name = f"sam_{args.exp_name}_method_{args.learn_method}_conf{int(args.sam_conf * 100)}_min_sev{args.min_dist}_segloss_IoU_trainloss_{args.lr}"
    if args.learn_method == 'qual':
        args.exp_name += f"_{args.learn_loss}"

    print(f'Running experiment {args.exp_name}')
    main(args, init=True if c == start_c else False)