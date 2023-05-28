"""
Inference time self-supervised learning
"""

import sys
import os
from pathlib import Path
import pickle
import cv2

import torch
import torchvision.transforms.v2 as tfms
import torchvision.transforms.v2.functional as F
from torchvision.transforms.v2.functional import to_pil_image, to_tensor
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from util.datasets_seg import get_pascal, get_test_dataloader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from eval import load_seg_model, load_tta_model, iou_loss
from util.voc_dataset_seg import sort_and_verify_sub

from tta import run_grabcut, denormalize_tensor

from torchmetrics.classification import BinaryJaccardIndex
from distortion import distortions


from arg_composition import get_segmentation_args

sys.path.append('..')

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


def mean_squared_error(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    return mse




def load_mae_results(args):
    """
    in case we want to load the images with the worst segmentation or something like that
    """
    mae_res_folder = f'results_num/{args.model_name}/maes_10/'
    with open(os.path.join(mae_res_folder, f'{args.dataset_name}.pkl'), 'rb') as f:  # Python 3: open(..., 'rb')
        mae_res = pickle.load(f)
    return mae_res


def main(args, thresh=0.4):
    def plot_results(name, img, gt, pred_seg, pred_grabcut, thresh=0.4,
                     save_path=None):
        plt.subplots(2, 3, figsize=(15, 10))

        # visualize
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title(f'Image {name}')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(gt, cmap='gray')
        plt.title(f'Ground truth')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(pred_seg, cmap='gray')
        plt.title(f'Prediction')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(pred_grabcut[0], cmap='gray')
        plt.title(f'Grabcut hard')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(pred_grabcut[1], cmap='gray')
        plt.title(f'Grabcut soft')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(pred_grabcut[2], cmap='gray')
        plt.title(f'Grabcut soft bg')
        plt.axis('off')

        if save_path is not None:
            plt.savefig(save_path)
        plt.show()

    # set dataset sub to B
    args.data_cls_sub = 'B'
    dataset = get_pascal(args, split='val')

    # reload model
    model_seg = load_seg_model(args)

    for idx in range(20):
        img, gt, cls, name = dataset[idx]


        gt = (gt > 0).int().squeeze()

        # add batch dimension
        im_tensor = img[None].to(device)

        model_seg.eval() # makes sure layers like batchnorm or dropout are in eval mode - doesn't prevent backprop

        with torch.no_grad():
            preds_seg = model_seg.forward_seg(im_tensor, inference=True)

        preds_seg = preds_seg.cpu().numpy().squeeze()

        # now test grabcut, initializing with predicted mask
        # img = denormalize(img)
        # img = to_pil_image(img)
        # img = np.array(img)
        # this shouldn't matter to grabcut
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        grabcuts = []

        # apply grabcut
        fg_thresh, fg_thresh_conf = 0.6, 0.8
        bg_thresh, bg_thresh_conf = 0.005, 0.001

        # initialize mask for grabcut
        mask = np.zeros_like(gt, dtype=np.uint8)
        mask[preds_seg > fg_thresh] = cv2.GC_PR_FGD
        mask[preds_seg > fg_thresh_conf] = cv2.GC_FGD
        mask[preds_seg < bg_thresh] = cv2.GC_PR_BGD
        mask[preds_seg < bg_thresh_conf] = cv2.GC_BGD

        img = denormalize(img)

        gc_mask, bgdModel, fgdModel = cv2.grabCut(np.array(to_pil_image(img)), mask, None, None, None, 5, cv2.GC_INIT_WITH_MASK)
        gc_pred = np.zeros_like(gt)
        gc_pred[gc_mask == cv2.GC_FGD] = 1 * 255
        gc_pred[gc_mask == cv2.GC_PR_FGD] = 0.8 * 255
        gc_pred[gc_mask == cv2.GC_PR_BGD] = 0.1 * 255

        grabcuts.append(gc_pred)
        # try variant with no confident regions

        # initialize mask for grabcut
        mask = np.zeros_like(gt, dtype=np.uint8)
        mask[preds_seg > 0.1] = cv2.GC_PR_FGD
        mask[preds_seg < 0.001] = cv2.GC_PR_BGD

        gc_mask, bgdModel, fgdModel = cv2.grabCut(np.array(to_pil_image(img)), mask, None, None, None, 5, cv2.GC_INIT_WITH_MASK)
        gc_pred = np.zeros_like(gt)
        gc_pred[gc_mask == cv2.GC_FGD] = 1 * 255
        gc_pred[gc_mask == cv2.GC_PR_FGD] = 0.8 * 255
        gc_pred[gc_mask == cv2.GC_PR_BGD] = 0.1 * 255

        grabcuts.append(gc_pred)
        # try variant with no confident bg regions

        # initialize mask for grabcut
        mask = np.zeros_like(gt, dtype=np.uint8)
        mask[preds_seg > 0.05] = cv2.GC_PR_FGD
        mask[preds_seg > 0.95] = cv2.GC_FGD
        mask[preds_seg < 0.001] = cv2.GC_PR_BGD

        gc_mask, bgdModel, fgdModel = cv2.grabCut(np.array(to_pil_image(img)), mask, None, None, None, 5, cv2.GC_INIT_WITH_MASK)
        gc_pred = np.zeros_like(gt)
        gc_pred[gc_mask == cv2.GC_FGD] = 1 * 255
        gc_pred[gc_mask == cv2.GC_PR_FGD] = 0.8 * 255
        gc_pred[gc_mask == cv2.GC_PR_BGD] = 0.1 * 255

        grabcuts.append(gc_pred)

        img = to_pil_image(img)
        img = np.array(img)

        # create the directory if it doesn't exist
        if not os.path.exists('gc_vis'):
            os.makedirs('gc_vis')
        save_path = f'gc_vis/{name}_{idx}.png'
        plot_results(name, img, gt, preds_seg, grabcuts, thresh=0.4, save_path=save_path)

        print()





def eval_seg(preds, gt, thresh=0.4):
    ious = []
    for pred in preds:
        pred = (pred > thresh).int()
        iou = (pred * gt).sum() / ((pred + gt) > 0).sum()
        ious.append(1 - iou)
    return ious


def im_to_vis(im):
    denorm_im = im * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                         torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    im = F.to_pil_image(denorm_im)
    return im


def small_eval_class(args, thresh=0.4, sub='B', seg_weight=0.5, rec_weight=1, cls_samples=20):
    """
    Evaluate segmentation on a small subset of the dataset - first 20 images for each class
    """

    folder = 'pascal_gc_small_class'
    os.makedirs(folder, exist_ok=True)

    _, cats = sort_and_verify_sub(sub)

    model_seg = load_seg_model(args)
    # try to only update encoder
    # model_seg.freeze_seg_decoder()
    model_seg.eval()

    seg_iou_losses, seg_iou_losses = [], []

    for cat in cats:
        #    set dataset sub to cat
        args.data_cls_sub = cat
        dataset = get_pascal(args, split='val')
        dataloader = get_test_dataloader(dataset, args)

        cat_seg_iou_losses, gc_cat_seg_iou_losses = [], []
        for i, (batch) in enumerate(tqdm(dataloader)):
            imgs, gts, loc_clses, names = batch
            imgs, gts = imgs.to(device), gts.to(device)

            gts = (gts > 0).int()

            with torch.no_grad():
                preds_seg = model_seg.forward_seg(imgs, inference=True)

            gc_preds = []
            for im_tensor, gt, pred in zip(imgs, gts, preds_seg):
                # im_tensor to denormalized np image for grabcut
                img = np.array(to_pil_image(denormalize(im_tensor[0].cpu())))


                img = denormalize_tensor(imgs[0].cpu())
                gc_pred = run_grabcut(np.array(to_pil_image(img)), pred.squeeze().cpu().numpy())
                gc_preds.append(gc_pred[None])
            gc_preds = torch.Tensor(gc_preds).to(device)

            gc_cat_seg_iou_losses.extend(eval_seg(gc_preds, gts, thresh))

            with torch.no_grad():
                preds_seg = model_seg.forward_seg(im_tensor, inference=True)

        seg_iou_losses.append(gc_cat_seg_iou_losses)

    # convert all results to np arrays
    seg_iou_losses = np.array(seg_iou_losses) * 100

    # print results (cat, n_ims, n_its), both for separate categories and aggregated, also save it to file
    log = ''
    save_name = f'opt_{args.tta_optim}_lr_{args.tta_lr}_bs_{args.tta_rec_batch_size}_its_{args.tta_iter_num}_rw_{rec_weight}_sw_{seg_weight}'
    # print the results in latex format for easy copy-paste
    log += f'Results for: {save_name}\n\n'
    log += 'seg_iou_loss\n'

    log += f'{np.mean(seg_iou_losses):.3f}\n'
    # now print the results for each category
    for cat, cat_seg_iou_losses in zip(cats, seg_iou_losses):
        log += f'{cat}:  {np.mean(cat_seg_iou_losses):.3f}\n'

    #     also print all iou results aggregated over samples and classes
    log += f'Iou results aggregated over samples and classes:\n'
    log += f'{np.mean(seg_iou_losses, axis=(0, 1))}\n'
    # now print the results for each category
    for cat, cat_seg_iou_losses in zip(cats, seg_iou_losses):
        log += f'{cat}: {np.mean(cat_seg_iou_losses, axis=0)}\n'

    print(log)
    # save log to txt file
    with open(f'{folder}/{save_name}.txt', 'w') as f:
        f.write(log)

    #     also save numpy array with results
    np.save(f'{folder}/{save_name}.npy', seg_iou_losses)


def small_eval_distorted():
    """
    Evaluate segmentation on a small subset of the dataset - first 20 images for each class
    """

    folder = 'pascal_gc_small_distorted'
    os.makedirs(folder, exist_ok=True)

    args = get_segmentation_args().parse_args()
    args.run_name = 'sweep_aspect2_pascal_A_SEG+REC_ps_16_p_ar'
    model_seg = load_seg_model(args)
    # try to only update encoder
    # model_seg.freeze_seg_decoder()
    model_seg.eval()

    seg_iou_losses, seg_iou_losses = [], []

    dataset = get_pascal(args, split='val')
    dataloader = get_test_dataloader(dataset, args)

    cat_seg_iou_losses, gc_cat_seg_iou_losses = [], []
    for i, (batch) in enumerate(tqdm(dataloader)):
        imgs, gts, loc_clses, names = batch
        imgs, gts = imgs.to(device), gts.to(device)

        gts = (gts > 0).int()

        with torch.no_grad():
            preds_seg = model_seg.forward_seg(imgs, inference=True)

        gc_preds = []
        for im_tensor, gt, pred in zip(imgs, gts, preds_seg):
            # im_tensor to denormalized np image for grabcut
            img = np.array(to_pil_image(denormalize(im_tensor[0].cpu())))

            img = denormalize_tensor(imgs[0].cpu())
            gc_pred = run_grabcut(np.array(to_pil_image(img)), pred.squeeze().cpu().numpy())
            gc_preds.append(gc_pred[None])
        gc_preds = torch.Tensor(gc_preds).to(device)

        gc_cat_seg_iou_losses.extend(eval_seg(gc_preds, gts, thresh))

        with torch.no_grad():
            preds_seg = model_seg.forward_seg(im_tensor, inference=True)

    seg_iou_losses.append(gc_cat_seg_iou_losses)

    # convert all results to np arrays
    seg_iou_losses = np.array(seg_iou_losses) * 100

    # print results (cat, n_ims, n_its), both for separate categories and aggregated, also save it to file
    log = ''
    save_name = f'opt_{args.tta_optim}_lr_{args.tta_lr}_bs_{args.tta_rec_batch_size}_its_{args.tta_iter_num}' \
                f'_rw_{1}_sw_{1}'


def denormalize(img):
    """
    Transform image-net normalized image to original [0, 1] range
    """
    return img * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]


if __name__ == '__main__':
    args = get_segmentation_args().parse_args()
    args.run_name = 'sweep_aspect2_pascal_A_SEG+REC_ps_16_p_ar'
    # small_eval_class(args)
    # small_eval_distorted(args)
    main(args)
