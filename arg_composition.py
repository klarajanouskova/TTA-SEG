import argparse
import os

import torch

local = not torch.cuda.is_available()


# TODO: create separate function for training (current train -> optim?), inference, tta, ...
def get_segmentation_args(inference=False, train=True):
    parser = argparse.ArgumentParser('Segmentation-finetune', add_help=False)
    parser.add_argument('--exp_name', default='seg_test', type=str)
    parser = get_segmentation_dataset_args(parser)
    parser = get_segmentation_model_args(parser)
    if inference:
        parser = get_inference_args(parser)
    if train:
        parser = get_segmentation_training_args(parser)
    parser = get_misc_args(parser)
    parser = get_tta_args(parser)
    parser = get_loss_training_args(parser)
    return parser


def get_segmentation_dataset_args(parser):
    if local:
        dataset_dir = '/Users/panda/VRG/TestTimeTraining/datasets'
    else:
        # dataset_dir = '/mnt/walkure_public/klara/datasets'
        dataset_dir = '/datagrid/TextSpotter/klara/datasets'

    # Dataset parameters
    parser.add_argument('--valid_part', default=500, type=float,
                        help='ratio/number of samples left for validation')
    parser.add_argument('--data_path', default=dataset_dir, type=str,
                        help='dataset path')
    parser.add_argument('--dataset', default='pascal', type=str, choices=['pascal', 'DUTS'])
    parser.add_argument('--data_cls_sub',
                        default='A',
                        type=str,
                        choices=['all', 'A', 'B', 'cat&dog'],
                        help='subset of classes to use for training, ie. A, B, or all')
    parser.add_argument('--preserve_aspect', default=1, type=int, choices=[0, 1],
                        help='preserve aspect ratio when resizing images')

    return parser


def get_model_args(parser):
    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_seg_conv_unet', type=str, metavar='MODEL',
                        help='Name of model to train, specifies ie patch size')

    parser.add_argument('--input_size', default=384, type=int,
                        help='images input size')
    parser.add_argument('--patch_size', default=16, type=int,
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
    parser.add_argument('--freeze_encoder', default=0, type=int, choices=[0, 1])

    return parser


def get_segmentation_model_args(parser):
    # Model parameters
    parser = get_model_args(parser)

    # segmentation specific params
    parser.add_argument('--unet_depth', default=2, type=int)

    return parser


def get_segmentation_training_args(parser):
    # orig: 64, we can run 8 on juliet per gpu for sure, 8 with 3 layer unet
    parser = get_training_args(parser)
    # segmentation specific params
    # STR: Structure Loss, FL: F-measure loss
    parser.add_argument('--loss', default='BCE+IoU', type=str, choices=['STR', 'FL', 'IoU', 'BCE', 'BCE+IoU'])
    parser.add_argument('--rec_weight', default=1., type=float, help='reconstruction loss weight')
    parser.add_argument('--seg_weight', default=1., type=float, help='segmentation loss weight')
    return parser


def get_training_args(parser):
    # orig: 64, we can run 8 on juliet per gpu for sure, 8 with 3 layer unet
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # optimization params
    parser.add_argument('--test', default=0, type=int, choices=[0, 1])
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=1,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=1, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='ckpts/mae_visualize_vit_base.pth',
                        help='finetune from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch, set automatically from checkpoint when resuming')

    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')

    return parser


def get_misc_args(parser):
    # Misc
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

    return parser


def get_inference_args(parser):
    parser.add_argument('--run_name', default='sweep_aspect2_pascal_A_SEG+REC_ps_16_p_ar',
                        help='identify the folder in output dir')
    return parser


def get_loss_training_args(parser):
    parser.add_argument('--learn_loss', default="l2", type=str, choices=["l1", "l2"])
    # quality or refinement estimation
    parser.add_argument('--learn_method', default="ref", type=str, choices=["ref", "qual"])
    parser.add_argument('--min_dist', default=0, type=int, help="distortion severity lower limit")
    parser.add_argument('--max_dist', default=5, type=int, help="distortion severity upper limit")
    parser.add_argument('--loss_use_gt', default=0, type=int, help="whether to compute loss wrt clean image mask, or the gt mask")
    parser.add_argument('--binarize_pred_gt', default=0, type=int, help="whether to binarize clean image mask before loss computation")
    parser.add_argument('--domain_shift_method', default='adversarial', type=str, help="which domain shift method to use", choices=["adversarial", "corruption"])
    return parser


def get_tta_args(parser):
    #  TODO replace ssl with tta in some future refactor
    parser.add_argument('--tta_n_ims', default=1, type=int, help="number of images to use in TTA for optimization")
    parser.add_argument('--tta_rec_batch_size', default=8, type=int)
    parser.add_argument('--tta_iter_num', default=10, type=int, help='number of iterations for SSL optimization')
    parser.add_argument('--tta_lr', default=1e-2, type=float, help='learning rate for SSL optimization')
    parser.add_argument('--loss_model', default="MaskLossNet", type=str, help='model class', choices=["MaskLossNet", "MaskLossDictNet", "MaskLossUnet"])
    # # l1 with distortions l2 was in eval for this run()
    parser.add_argument('-tta_iou_model_run', default="deep_loss_adv_clean_gt_soft_method_qual_min_sev0_segloss_IoU_trainloss_0.0005_l1", type=str, help='weight decay for SSL optimization')
    parser.add_argument('-tta_ref_model_run', default="deep_loss_adv_clean_gt_soft_method_ref_min_sev0_segloss_IoU_trainloss_0.0005", type=str, help='weight decay for SSL optimization')
    parser.add_argument('--tta_optim', default="sgd", type=str, choices=["adam", "sgd"])
    # ref - refinement, rec - reconstruction, l1 - l1 predictor. Currently l1 and ref are run with rec by default,
    # should be changed to support combinations
    parser.add_argument('--tta_method', default='ref', type=str, choices=['gc', 'rec', 'l1', 'ref', 'l2'])
    parser.add_argument('--tta_freeze_rec', default=1, type=int, choices=[0, 1])
    parser.add_argument('--tta_grad_clip', default=-1, type=float)
    parser.add_argument('--tta_ref_post', default=0, type=int, choices=[0, 1], help="whether to use postprocessing with "
                                                                                    "deep mask refinement after tta")
    return parser