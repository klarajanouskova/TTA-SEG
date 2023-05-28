# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.functional as F


import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory
from timm.models.layers import trunc_normal_

import wandb


import util.lr_decay as lrd
import util.misc as misc
from util.datasets_reconstruct import build_dataset, get_dataloaders
from util.datasets_reconstruct import SingleClassImageFolder
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch, evaluate

local = not torch.cuda.is_available()


def get_args_parser():
    if local:
        dataset_dir = '/Users/panda/Technion/datasets'
    else:
        dataset_dir = '/home/klara/datasets/'


    parser = argparse.ArgumentParser('TTA pre-training', add_help=False)
    parser.add_argument('--exp_name', default='test', type=str)
    # orig: 64, we can run 12/home/klara/datasets8 on juliet per gpu for sure
    parser.add_argument('--batch_size', default=150, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=100, type=int)
    # try increasing this
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    # Dataset parameters
    parser.add_argument('--valid_part', default=0.05, type=float,
                        help='ratio/number of samples left for validation')
    # parser.add_argument('--data_path', default=os.path.join(dataset_dir, 'VOC/VOCdevkit/VOC2012/'), type=str,
    #                     help='dataset path')
    parser.add_argument('--data_path', default=dataset_dir, type=str,
                        help='dataset path')
    # a bit of a hack so that we can use a simple wrapper around the ImageFolder dataset, TODO rewrite
    parser.add_argument('--dataset', default='DUTS', type=str, choices=['DUTS', 'VOC', 'COCO', 'ImageNet'],
                        help='the name of the folder containing images in the dataset root')
    # parser.add_argument('--class_folder', default='JPEGImages', type=str,
    #                     help='the name of the folder containing images in the dataset root')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # default finetune setup
    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # parser.add_argument('--layer_decay', type=float, default=0.75,
    #                     help='layer-wise lr decay from ELECTRA/BEiT')
    # parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0')
    #
    # parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
    #                     help='epochs to warmup LR')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='ckpts/mae_visualize_vit_base.pth',
                        help='finetune from checkpoint')

    parser.add_argument('--output_dir', default='./out',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint to continue training')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch, set automatically from checkpoint when resuming')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args, init=True):

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
    data_loader_train, data_loader_val = get_dataloaders(args, global_rank, num_tasks)

    # define the model
    model = models_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    if global_rank == 0:
        # has to be done before the summarywriter is created!
        wandb.init(project="TTA-finetune", entity="klara", name=args.exp_name, reinit=True)
        wandb.config = vars(args)
        wandb.watch(model)
        if args.output_dir is not None:
            args.log_dir = os.path.join(args.output_dir, args.exp_name, 'logs')
            os.makedirs(args.log_dir, exist_ok=True)
            if init:
                wandb.tensorboard.patch(root_logdir=args.log_dir, pytorch=True, save=False)
            log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # load pretrained checkpoint
    if args.finetune and not args.eval_saliency:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # TODO we might want to remove this
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    # build optimizer with layer-wise lr decay (lrd), early layers have low lr, later layers higher
    # pos. enc. and cls token are not here/not learnable, TODO why?
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # necessary for mixed precision training
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss_train, min_loss_val = np.inf, np.inf

    if data_loader_val:
        # evaluate before we start training
        val_stats = evaluate(data_loader_val, model, device, args)
        print(f"Validation loss before training: {val_stats['loss']:.1f}%")
        min_loss_val = min(min_loss_val, val_stats["loss"])

    if global_rank == 0:
        wandb.log({"val/min_loss": min_loss_val})

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        if data_loader_val:
            val_stats = evaluate(data_loader_val, model, device, args)
            print(f"Loss of the network on the validation images: {val_stats['loss']:.1f}%")

        if data_loader_val:
            if min_loss_val > val_stats["loss"]:
                min_loss_val = val_stats["loss"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, best=True)
        min_loss_train = min(min_loss_train, train_stats["loss"])

        print(f'Min loss val: {min_loss_val:.2f}%')
        print(f'Min loss train: {min_loss_train:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, args.exp_name, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if global_rank == 0:
            wandb.log({"val/min_loss": min_loss_val})
            wandb.log({"val/loss": val_stats["loss"]})
            wandb.run.summary["val/min_loss"] = min_loss_val
            wandb.log({"train/min_loss": min_loss_val})
            wandb.run.summary["train/min_loss"] = min_loss_val


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    wandb.finish()


if __name__ == '__main__':
    # Launch: python -m torch.distributed.launch --nproc_per_node=4 main_finetune_self.py --world_size 4 --model mae_vit_base_patch16
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    blrs = [1e-5]
    acc_iters = [1]
    c = 0
    start_c = 0
    misc.init_distributed_mode(args)
    for accum_iter in acc_iters:
        for blr in blrs:
            if c < start_c:
                c += 1
                continue
            args.blr = blr
            args.accum_iter = accum_iter
            args.exp_name = f"{args.dataset}_tr_256_blr_{blr}_accum_{accum_iter}"
            print(f'Running experiment {args.exp_name}')
            main(args, init=True if c == start_c else False)
            c += 1