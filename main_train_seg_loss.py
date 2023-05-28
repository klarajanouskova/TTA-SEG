# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import wandb

import util.misc as misc
from util.datasets_seg import get_dataloaders

from models_loss import MaskLossNet, MaskLossUnet

from engine_train_seg_loss import train_one_epoch, evaluate
from arg_composition import get_segmentation_args

from eval import load_seg_model# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import datetime

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import wandb

import util.misc as misc
from util.datasets_seg import get_dataloaders

from models_loss import MaskLossNet, MaskLossUnet

from engine_train_seg_loss import train_one_epoch, evaluate
from arg_composition import get_segmentation_args

from eval import load_seg_model

local = not torch.cuda.is_available()


def get_model(args):
    #     ["l1", "l2", "ref-l1", "ref-l2"]
    if args.learn_method == 'qual':
        return MaskLossNet()
    elif args.learn_method == 'ref':
       return MaskLossUnet()



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
    data_loader_train, data_loader_val = get_dataloaders(args, global_rank, num_tasks)

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
    model_seg = load_seg_model(args, pick='best')

    model_loss.to(device)
    model_seg.eval()

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
    args = get_segmentation_args(inference=True).parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    c = 0
    start_c = 0
    args.exp_name = f"{args.exp_name}_method_{args.learn_method}_min_sev{args.min_dist}_segloss_IoU_trainloss_{args.lr}"
    if args.learn_method == 'qual':
        args.exp_name += f"_{args.learn_loss}"

    print(f'Running experiment {args.exp_name}')
    main(args, init=True if c == start_c else False)

local = not torch.cuda.is_available()


def get_model(args):
    #     ["l1", "l2", "ref-l1", "ref-l2"]
    if args.learn_method == 'qual':
        return MaskLossNet()
    elif args.learn_method == 'ref':
       return MaskLossUnet()



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
    data_loader_train, data_loader_val = get_dataloaders(args, global_rank, num_tasks)

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
    model_seg = load_seg_model(args, pick='best')

    model_loss.to(device)
    model_seg.eval()

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
    args = get_segmentation_args(inference=True).parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    c = 0
    start_c = 0
    args.exp_name = f"{args.exp_name}_method_{args.learn_method}_min_sev{args.min_dist}_segloss_IoU_trainloss_{args.lr}"
    if args.learn_method == 'qual':
        args.exp_name += f"_{args.learn_loss}"

    print(f'Running experiment {args.exp_name}')
    main(args, init=True if c == start_c else False)