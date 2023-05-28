"""
Code: https://github.com/mczhuge/ICON
Author: mczhuge
Desc: Core code for evaluationg SOD
"""

from tqdm import tqdm
import icon_metrics as M
import pickle
import argparse

import sys

import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision.transforms as tfms
import torchvision.transforms.functional as F

from util.icon_dataset import Data, Transformation, Resize
import models_mae
from util.icon_dataset import get_test_dataloader
from util.voc_dataset_seg import VOCSegmentationSubFgBg, HBBoxTransform
from inference_tta_old import optimize_im
from icon_metrics import *
from eval import load_seg_model

sys.path.append('..')
import models_mae, models_conv_mae

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


if local:
    # used for icon
    # dataset_dir = '/Users/panda/Technion/datasets'
    # used for pascal
    dataset_dir = '/Users/panda/datasets'
else:
    dataset_dir = '/mnt/walkure_public/klara/datasets'


datasets = ['DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SOD']
DUTS_MEAN = np.array([102.94, 118.90, 124.55])
DUTS_STD = np.array([57.50, 55.97, 56.77])


def mean_squared_error(pred, gt):
    mse = ((pred - gt) ** 2).mean()
    return mse


def show_image(image, title=''):
    # image is [H, W, 3]
    if len(image.shape) > 2 and image.shape[2] == 3:
        plt.imshow(torch.clip((image * DUTS_STD + DUTS_MEAN), 0, 255).int())
    else:
        plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return






DUTS_MEAN = torch.Tensor([102.94, 118.90, 124.55]).to(device)
DUTS_STD = torch.Tensor([57.50, 55.97, 56.77]).to(device)


def denorm(im):
    im = im * DUTS_STD[:, None, None] + DUTS_MEAN[:, None, None]
    return im


def norm(im):
    im = (im - DUTS_MEAN[:, None, None]) / DUTS_STD[:, None, None]
    return im


def eval_saliency(args):
    args.batch_size = 1 if args.tta_iter_num > 0 else args.batch_size
    out_path = f'results_num/'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(out_path, f'{args.model_name}.txt')
    with open(out_file, 'w+') as log:
        # eval(model, args, log)
        # for thresh in np.arange(0.1, 1, 0.1):
        for thresh in [0.1]:
            log.write('-' * 60)
            log.write(f'Evaluating threshold {thresh}\n\n')
            print('-' * 60)
            print(f'Evaluating threshold {thresh}\n\n')
            # for dataset_name in ['DUT-OMRON', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'SOD', 'DUTS']:
            #     for dataset_name in ['SOD']:
            for dataset_name in ['PASCAL-S-F']:
                if args.visualize:
                    Path(f'results_vis/{args.model_name}/{dataset_name}').mkdir(parents=True, exist_ok=True)
                data_loader_test = get_test_dataloader(args, dataset_name)

                if args.tta_iter_num > 0:
                    # for lr in [5e-3]:
                    #     for mr in [0.5, 0.75]:
                    #         gauss_noise_fun = lambda x: norm((denorm(x) / 255 + torch.randn_like(x) * 0.2).clip(0, 1) * 255)
                    #         eval_dataset_ssl(data_loader_test, dataset_name, args, log, thresh=thresh, lr=lr, mask_ratio=mr,
                    #                          corruption_fun=gauss_noise_fun)

                    gauss_noise_fun = lambda x: norm((denorm(x) / 255 + torch.randn_like(x) * 0.2).clip(0, 1) * 255)

                    for lr in [5e-3]:
                        for mr in [0.75]:
                            eval_dataset_ssl(data_loader_test, dataset_name, args, log, thresh=thresh, lr=lr, mask_ratio=mr,
                                             corruption_fun=gauss_noise_fun)
                else:
                    model = load_seg_model(args)
                    eval_dataset(model, data_loader_test, dataset_name, args, log, thresh=thresh)


class ToTensorUnscaled:
    def __init__(self, dtype=torch.float):
        self.dtype = dtype

    def __call__(self, image):
        image = F.convert_image_dtype(torch.tensor(image), self.dtype)
        if len(image.shape) == 2:
            image = image.unsqueeze(-1)
        image = torch.einsum('hwc->chw', image)
        return image


def eval_pascal(args, sub="cat&dog"):
    out_path = f'results_num/{args.model_name}'
    Path(out_path).mkdir(parents=True, exist_ok=True)
    dataset_name = f'pascal_{sub}'
    out_file = os.path.join(out_path, f'{dataset_name}.txt')

    root = os.path.join(dataset_dir, 'voc')
    # add some space around the objects if possible
    bbox_trans = HBBoxTransform(range=(0.2, 0.2))
    # sub = 'all'
    im_transform = tfms.Compose([Transformation(train=False, size=args.input_size)])
    # Eventually, the resize here should be removed and predictions should be resized to match gt shape
    mask_transform = tfms.Compose([Resize(args.input_size, args.input_size)])

    dataset = VOCSegmentationSubFgBg(root=root,
                                     sub=sub,
                                     transform=im_transform,
                                     target_transform=mask_transform,
                                     bbox_transform=bbox_trans)
    batch_size = 1 if args.tta_iter_num > 0 else args.batch_size
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, drop_last=False)

    if args.visualize:
        Path(f'results_vis/{args.model_name}/{dataset_name}').mkdir(parents=True, exist_ok=True)

    with open(out_file, 'w+') as log:
        for thresh in [0.05]:
            if args.tta_iter_num > 0:
                for lr in [1e-2]:
                    for mr in [0.75, 0.5]:
                        gauss_noise_fun = lambda x: (denorm(x) + torch.randn_like(x) * 0.2)
                        # TODO needs non-negative values
                        poisson_noise_fun = lambda x: (x + torch.poisson(x * 0.9) / 0.9).clamp(0, 1)
                        gauss_fun = lambda x: torch.einsum('chw -> hwc', F.gaussian_blur(torch.einsum('hwc -> chw', x), kernel_size=5, sigma=3))
                        contrast_fun = lambda x: torch.einsum('chw -> hwc', F.adjust_contrast(torch.einsum('hwc -> chw', x), 2))
                        brightness_fun = lambda x: torch.einsum('chw -> hwc', F.adjust_brightness(torch.einsum('hwc -> chw', x), 2))
                        eval_dataset_ssl(dataloader, dataset_name, args, log, thresh=thresh, lr=lr, mask_ratio=mr, corruption_fun=gauss_noise_fun)
            else:
                model = load_seg_model(args)
                eval_dataset(model, dataloader, dataset_name, args,  log, thresh=thresh)


def eval_dataset(model, data_loader_test, dataset_name, args, log, thresh=None):
    FM = M.Fmeasure_and_FNR()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    mae_names, rec_errs = [], []
    with torch.no_grad():
        for i, (batch) in enumerate(tqdm(data_loader_test)):
            if len(batch) == 3:
                imgs, gts, names = batch
            else:
                # pascal
                imgs, gts, _, names = batch
            imgs = imgs.to(device)
            # gts = gts.to(device)
            preds_seg = model.forward_seg(imgs)
            preds_seg = preds_seg.detach().cpu().numpy()
            preds_seg = (preds_seg > thresh).astype(int) if thresh is not None else preds_seg
            gts = gts.detach().cpu().numpy()

            loss_rec, preds_rec, masks = model.forward_rec(imgs)
            rec_errs.extend([mean_squared_error(pred_rec, img).item() for pred_rec, img in zip(preds_rec, imgs)])

            for pred, gt, im, name in zip(preds_seg, gts, imgs, names):
                pred, gt = pred.squeeze(), gt.squeeze()
                if args.visualize and i < 1:
                    plt.subplot(1, 3, 1)
                    show_image(torch.einsum('chw->hwc', im.cpu()), title='Image')
                    plt.subplot(1, 3, 2)
                    show_image(gt, title='Ground Truth')
                    plt.subplot(1, 3, 3)
                    show_image(pred, title='Prediction')
                    plt.savefig(f'results_vis/{args.model_name}/{dataset_name}/{name}.pdf', bbox_inches='tight')

                FM.step(pred=pred, gt=gt)
                WFM.step(pred=pred, gt=gt)
                SM.step(pred=pred, gt=gt)
                EM.step(pred=pred, gt=gt)
                MAE.step(pred=pred, gt=gt)
                # FNR.step(pred=pred, gt=gt)

                mae_names.append(name)

    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]

    res_folder = f'results_num/{args.model_name}/maes_{int(thresh * 100)}/'
    Path(res_folder).mkdir(parents=True, exist_ok=True)
    save_dict = {"MAEs": np.array(MAE.maes), "names": mae_names, 'REC_ERRs': np.array(rec_errs)}
    with open(os.path.join(res_folder, f'{dataset_name}.pkl'), 'wb') as f:
        pickle.dump(save_dict, f)

    print(
        '\t', dataset_name,
        '\n',
        # 'Attribute:', args.attr, '||',
        '\t Smeasure:', sm.round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'TTA:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'maxFm:', fm['curve'].max().round(3), '; ',
        'fnr:', fnr.round(3), "\n\n",
        sep=''
    )
    log_items = ['\t', dataset_name,
                 '\n',
                 # 'Attribute:', args.attr, '||',
                 '\t Smeasure:', sm.round(3), '; ',
                 'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
                 'wFmeasure:', wfm.round(3), '; ',
                 'TTA:', mae.round(3), '; ',
                 'adpEm:', em['adp'].round(3), '; ',
                 'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
                 'adpFm:', fm['adp'].round(3), '; ',
                 'meanFm:', fm['curve'].mean().round(3), '; ',
                 'maxFm:', fm['curve'].max().round(3), '; ',
                 'fnr:', fnr.round(3), "\n\n"]
    logs = ' '.join(str(e) for e in log_items)
    log.write(logs)


def eval_dataset_ssl(data_loader_test, dataset_name, args, log, thresh=None, lr=1e-2, mask_ratio=0.75, corruption_fun=None):
    def reload_model():
        model = load_seg_model(args)
        model.eval()
        model.freeze_decoder()
        return model
    names, mses_rec, maes_seg, ious_seg = [], [], [], []
    for i, (batch) in enumerate(tqdm(data_loader_test)):
        if len(batch) == 3:
            img, gt, name = batch[0][0], batch[1][0], batch[2][0]
        else:
            # pascal
            img, gt, _, name = batch[0][0], batch[1][0], batch[2][0], batch[3][0]
        #     copy image batch_size time
        img = img.to(device)
        # gts = gts.to(device)

        model = reload_model()
        if corruption_fun:
            corr_img = corruption_fun(img)
        else:
            corr_img = img
        preds_seg, preds_rec, im_mses_rec = optimize_im(torch.einsum('chw->hwc', corr_img), model, num_it=args.tta_iter_num, thresh=thresh, bs=args.batch_size,
                                           debug=False, lr=lr, mask_ratio=mask_ratio, clean_im=torch.einsum('chw->hwc', img))
        preds_seg, preds_rec = np.array(preds_seg), np.array(preds_rec)

        gt = gt.detach().cpu().numpy()
        gt = (np.einsum('chw->hwc', gt) > 0).astype(int).squeeze()

        # rec. can't be reconstructed here now because we would need to take care of the masked regions
        im_maes_seg, im_ious_seg = eval_preds(preds_seg, gt)

        names.append(name)
        maes_seg.append(im_maes_seg)
        mses_rec.append(im_mses_rec)
        ious_seg.append(im_ious_seg)

    noise_str = '_noise_cleanseg' if corruption_fun else ''
    results = pd.DataFrame({'name': names, 'mae_seg': maes_seg, 'mse_rec': mses_rec, 'iou_seg': ious_seg})
    res_folder = f'results_num/{args.model_name}/{dataset_name}/{int(thresh * 100)}/'
    Path(res_folder).mkdir(parents=True, exist_ok=True)
    results.to_csv(os.path.join(res_folder, f'{lr}_{mask_ratio}{noise_str}.csv'), index=False)
    print()


def eval_preds(preds_seg, gt):
    maes_seg, ious_seg = [], []
    for pred_seg in preds_seg:
        mae_seg = MAE.cal_mae(pred_seg, gt)
        iou_seg = IoU.cal_iou(pred_seg, gt)

        maes_seg.append(mae_seg)
        ious_seg.append(iou_seg)

    return maes_seg, ious_seg


def main():
    args = parser.parse_args()

    if local:
        args.batch_size = 4
    eval_saliency(args)
    # for sub in ['pottedplant&aeroplane']:
    #     eval_pascal(args, sub)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    if not local:
        parser.add_argument('--model_name', type=str, default='unet3_DUTS_norm_SEG+REC_lr_5e-05_BCE+IoU_inp_384', help='name of the model to eval')
    else:
        parser.add_argument('--model_name', type=str, default='unet3_bce_iou', help='name of the model to eval')
    parser.add_argument('--model_pick', type=str, default='best', choices=['best', 'last'])
    parser.add_argument('--model', type=str, default='mae_vit_base_patch16_seg_conv_unet')
    parser.add_argument('--out_sub', type=str, default='no_thresh')
    parser.add_argument('--unet_depth', default=3, type=int)
    parser.add_argument('--data_path', default=dataset_dir, type=str,
                        help='dataset path')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=384, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--ssl_iter_num', default=20, type=int, help='number of iterations for SSL optimization')
    parser.add_argument('--visualize', default=0, type=int)

    main()