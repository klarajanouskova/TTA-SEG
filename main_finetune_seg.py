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
import time
import json
from collections import OrderedDict

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm

# let's try to get away with this, https://github.com/facebookresearch/mae/issues/17 saysi t should be fine
# assert timm.__version__ == "0.3.2"  # version check

import wandb


import util.lr_decay as lrd
import util.misc as misc
from util.datasets_seg import get_dataloaders
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_conv_mae

from engine_finetune_seg import train_one_epoch, evaluate
from arg_composition import get_segmentation_args

local = not torch.cuda.is_available()


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

    # define the model
    model = models_conv_mae.__dict__[args.model](norm_pix_loss=args.norm_pix_loss,
                                                 drop_path_rate=args.drop_path,
                                                 img_size=args.input_size,
                                                 unet_depth=args.unet_depth,
                                                 patch_size=args.patch_size)

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

    # load pretrained checkpoint
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        new_ckpt_dict = OrderedDict()
        # put encoder/decoder prefix where necessary
        for k, v in checkpoint_model.items():
            assigned = False
            if k in state_dict and checkpoint_model[k].shape == state_dict[k].shape:
                new_ckpt_dict[k] = v
                assigned = True
            else:
                if 'encoder.' + k in state_dict:
                    # relax restrictions for positional embeddings
                    if 'pos_embed' in k or checkpoint_model[k].shape == state_dict['encoder.' + k].shape:
                        new_ckpt_dict['encoder.' + k] = v
                        assigned = True
                if 'decoder_rec.' + k in state_dict:
                    if 'pos_embed' in k or checkpoint_model[k].shape == state_dict['decoder_rec.' + k].shape:
                        new_ckpt_dict['decoder_rec.' + k] = v
                        assigned = True
            if not assigned:
                print("Warning: %s weight was not assigned in the new model" % k)


        # interpolate position embedding - TODO check if it helps
        interpolate_pos_embed(model, new_ckpt_dict)

        # load pre-trained model
        msg = model.load_state_dict(new_ckpt_dict, strict=False)
        print(msg)

    model.to(device)

    if args.freeze_encoder:
        model.freeze_encoder()

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
    elif not local:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    if global_rank == 0:
        wandb.watch(model_without_ddp)

    # following timm: set wd as 0 for bias and norm layers
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay
    )
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # necessary for mixed precision training - scales the loss so that gradients do not vanish, then scales back
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss_train, min_loss_val = np.inf, np.inf

    if data_loader_val:
        # evaluate before we start training
        val_stats = evaluate(data_loader_val, model, device, args, log_ims=True, pre='val', first_val=True)
        print(f"Validation loss before training: {val_stats['val/loss']:.1f}")
        min_loss_val = min(min_loss_val, val_stats["val/loss"])
        misc.save_model(
            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=0, best=True)

        if global_rank == 0:
            wandb.log({"val/min_loss": min_loss_val})
            wandb.log(val_stats)
            wandb.run.summary["val/min_loss"] = min_loss_val


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            use_lr_sched=False,
            args=args
        )

        min_loss_train = min(min_loss_train, train_stats["train/loss"])

        print(f'Min loss train: {min_loss_train:.2f}%')

        if data_loader_val:
            val_stats = evaluate(data_loader_val, model, device, args, log_ims=True, pre='val')
            print(f"Loss of the network on the validation images: {val_stats['val/loss']:.1f}%")

            if min_loss_val > val_stats["val/loss"]:
                min_loss_val = val_stats["val/loss"]
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, best=True)
            val_stats["val/min_loss"] = min_loss_val

            print(f'Min loss val: {min_loss_val:.2f}%')


        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            #     TODO images can't be json dumped
            # log_stats = {**{f'{k}': v for k, v in train_stats.items()},
            #              **{f'{k}': v for k, v in val_stats.items()},
            #              'epoch': epoch} if val_stats else {**{f'{k}': v for k, v in train_stats.items()},
            #                                                 'epoch': epoch}
            # with open(os.path.join(args.output_dir, args.exp_name, "log.txt"), mode="a", encoding="utf-8") as f:
            #     f.write(json.dumps(log_stats) + "\n")

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

    misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, last=True)

    wandb.finish()


if __name__ == '__main__':
    # Launch: python -m torch.distributed.launch --nproc_per_node=4 main_finetune_self.py --world_size 4 --model mae_vit_base_patch16
    args = get_segmentation_args().parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    misc.init_distributed_mode(args)

    # lrs = [1e-4]
    # layer_decays = [0.65]
    c = 0
    start_c = 0
    # for accum_iter in acc_iters:
    #     for lr in lrs:
    #         if c < start_c:
    #             c += 1
    #             continue
    #         args.lr = lr
    #         args.accum_iter = accum_iter
    args.exp_name = f"{args.exp_name}{args.unet_depth}_{args.dataset}_{args.data_cls_sub}_SEG+REC_ps_{args.patch_size}"
    if args.freeze_encoder:
        args.exp_name += "_freeze_enc"
    if args.preserve_aspect:
        args.exp_name += "_p_ar"

    print(f'Running experiment {args.exp_name}')
    main(args, init=True if c == start_c else False)
            # c += 1