from copy import deepcopy
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import MetricCollection
from torchvision.transforms.v2.functional import to_pil_image, to_tensor
import torch
from tqdm import tqdm
import numpy as np

from pathlib import Path

from util.misc import MetricLogger
from util.datasets_seg import get_pascal, get_test_dataloader
from util.voc_dataset_seg import CLS2CAT

from eval import load_seg_model
from arg_composition import get_segmentation_args

from tta import run_grabcut, denormalize_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def valid_pascal(model, dataloader, threshold=0.4, log_path="pascal_results/out.txt", grabcut=False):
    model.eval()
    n_cls = len(dataloader.dataset.cats)
    iou_threshs = torch.arange(0.5, 1, 0.05).tolist()
    pred_threshs = torch.arange(0.1, 1, 0.1).tolist()
    map_metric = MeanAveragePrecision(num_classes=n_cls, iou_type="segm", bbox_format="xywh", class_metrics=True, iou_thresholds=iou_threshs).to(device)
    # create torchmetrics collection of iou for multiple threshold values
    iou_metrics = {cls: MetricCollection({f'iou_{int(t * 100)}': BinaryJaccardIndex(threshold=t) for t in pred_threshs}).to(device) for cls in range(n_cls)}

    with torch.no_grad():
        for i, (batch) in enumerate(tqdm(dataloader)):
            imgs, gts, loc_clses, names = batch
            imgs, gts = imgs.to(device), gts.to(device)
            # gts = gts.to(device)
            preds_seg = model.forward_seg(imgs, inference=True)
            if grabcut:
                gc_preds = []
                for pred in preds_seg:
                    img = denormalize_tensor(imgs[0].cpu())
                    gc_pred = run_grabcut(np.array(to_pil_image(img)), pred.squeeze().cpu().numpy())
                    gc_preds.append(gc_pred[None])
                preds_seg = torch.Tensor(gc_preds).to(device)
            preds = [{
                # we don't care about bboxes, so let's consider the whole image as a bbox
                "boxes": torch.Tensor([(0, 0, gt.shape[2], gt.shape[1])]).to(device),
                "labels": loc_cls[None].to(device),
                "scores": torch.ones(1).to(device),
                "masks": pred > threshold
            } for pred, gt, loc_cls in zip(preds_seg, gts, loc_clses.argmax(1))]
            targets = [{
                # we don't care about bboxes, so let's consider the whole image as a bbox
                "boxes": torch.Tensor([(0, 0, gt.shape[2], gt.shape[1])]).to(device),
                "labels": loc_cls[None].to(device),
                "masks": gt > 0
            } for pred, gt, loc_cls in zip(preds_seg, gts, loc_clses.argmax(1))]
            map_metric.update(preds, targets)
            for cls in range(n_cls):
                iou_metrics[cls].update(preds_seg[loc_clses.argmax(1) == cls], gts[loc_clses.argmax(1) == cls])


    # categories = [CLS2CAT[dataloader.dataset.su
    #     # categories = [CLS2CAT[dataloader.dataset.subb2globcls[c]] for c in range(n_cls)]
    map_result = map_metric.compute()
    iou_results = {cls: iou_metrics[cls].compute() for cls in range(n_cls)}

    results = {'map': map_result['map'], 'map_50': map_result['map_50'], 'map_75': map_result['map_75']}
    verbose_results = {}

    # add per-class iou according to best overall threshold
    best_t = pred_threshs[torch.Tensor([[iou_results[cls][f'iou_{int(t * 100)}'] for cls in range(n_cls)] for t in pred_threshs]).mean(1).argmax()]
    for cls in range(n_cls):
        results[f'iou_{CLS2CAT[dataloader.dataset.sub2globcls[cls]]}'] = iou_results[cls][f'iou_{int(best_t * 100)}']
    # also add the mIoU according to the best overall threshold
    results[f'mIoU'] = torch.stack([iou_results[cls][f'iou_{int(best_t * 100)}'] for cls in range(n_cls)]).mean()

    for cls in range(n_cls):
        verbose_results[f'map_{CLS2CAT[dataloader.dataset.sub2globcls[cls]]}'] = map_result['map_per_class'][cls]

    # add mIoU per threshold
    for t in pred_threshs:
        verbose_results[f'mIoU_{int(t * 100)}'] = torch.stack([iou_results[cls][f'iou_{int(t * 100)}'] for cls in range(n_cls)]).mean()

        for cls in range(n_cls):
            verbose_results[f'iou_{CLS2CAT[dataloader.dataset.sub2globcls[cls]]}_{int(t * 100)}'] = iou_results[cls][f'iou_{int(t * 100)}']
    # add results to verbose_results
    verbose_results.update(results)

    cats = [CLS2CAT[dataloader.dataset.sub2globcls[cls]] for cls in range(n_cls)]
    with open(log_path, 'w') as f:
        f.write('IoU (first row) and mAP (second row) per class, and mIoU and mAP overall\n')

        f.write("\n\n")

        # print the class names separated by '&'
        f.write(f"{' & '.join(cats)}\n")
        # print the iou per class for best threshold
        cls_ious = [results[f'iou_{cat}'] for cat in cats]
        f.write(f"{' & '.join([f'{iou * 100:.2f}' for iou in cls_ious])}\n")
        # print tha maps per class
        cls_maps = [verbose_results[f'map_{cat}'] for cat in cats]
        f.write(f"{' & '.join([f'{map * 100:.2f}' for map in cls_maps])}\n")


        f.write("\n\n")


        # print IoU per class for different thresholds
        f.write(f"{' & '.join(cats)}\n")
        for t in pred_threshs:
            f.write(f"IoU for threshold {t}\n")
            cls_ious = [verbose_results[f'iou_{cat}_{int(t * 100)}'] for cat in cats]
            f.write(f"{' & '.join([f'{iou * 100:.2f}' for iou in cls_ious])}\n")

        f.write("\n\n")

        # print the mAP overall, mAP 50 and mAP 75
        f.write('mAP overall:')
        val = verbose_results['map']
        f.write(f": {val * 100:.2f}\n")
        f.write('mAP 50:')
        val = verbose_results['map_50']
        f.write(f": {val * 100:.2f}\n")
        f.write('mAP 75:')
        val = verbose_results['map_75']
        f.write(f": {val * 100:.2f}\n")

        # also mIoU for best threshold
        f.write('mIoU:')
        val = verbose_results['mIoU']
        f.write(f": {val * 100:.2f}\n")

        f.write("\n\n")

        # print the thresholds separated by '&', then the mIoU per threshold
        f.write("mIoU per threshold")

        f.write("\n\n")

        f.write(f"{' & '.join([f'{int(t * 100)}' for t in pred_threshs])}\n")
        thresh_mious = [verbose_results[f'mIoU_{int(t * 100)}'] for t in pred_threshs]
        f.write(f"{' & '.join([f'{mIoU * 100:.2f}' for mIoU in thresh_mious])}\n")

        # for k, v in verbose_results.items():
        #     f.write(f"{k}: {v}\n")
    return results


def test_pascal_subsets(model, args, subsets=['B', 'A'], pre=None, save_name=None, grabcut=False):
    if save_name is None:
        save_name = args.run_name + '_best'
        if grabcut:
            save_name += '_grabcut'
    tmp_args = deepcopy(args)
    results = {}

    # set pascal as dataset
    pre = '' if pre is None else f'{pre}/'
    for subset in subsets:
        tmp_args.data_cls_sub = subset

        dataset = get_pascal(tmp_args, split='val')
        dataloader = get_test_dataloader(dataset, tmp_args)
        out_dir = f"pascal_results/{subset}"
        out_file = f"{out_dir}/{save_name}.txt"
        # make output directory if it doesn't exist
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        print(f"Testing on pascal {subset} subset, model name: {save_name}")
        result_dict = valid_pascal(model, dataloader, log_path=out_file, grabcut=grabcut)
        #   add subset  prefix to all keys
        result_dict = {f'{pre}{subset}/{k}': v for k, v in result_dict.items()}
        #  add to results
        results.update(result_dict)
    return results


if __name__ == '__main__':
    args = get_segmentation_args(inference=True).parse_args()
    runs = [
        'sweep_aspect2_pascal_B_SEG+REC_ps_16_p_ar',
        'sweep_aspect2_pascal_A_SEG+REC_ps_16_p_ar'
    ]
    for run in runs:
        args.run_name = run
        model = load_seg_model(args, pick='best')
        args.batch_size = 16
        test_pascal_subsets(model, args, grabcut=True)