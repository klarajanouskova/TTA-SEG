import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append('..')
sys.path.append('modeling')

from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import MetricCollection

from cityscapes_ext import PointCityscapes, PointCityscapesRain, PointCityscapesFog, cityscapes_root
from mask_generator import SamPointMaskGenerator
from modeling.build_sam import sam_model_registry
from sam_demo import show_anns, show_points

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_sam(path=None):
    if path is None:
        path = 'sam_vit_b.pth'
    sam = sam_model_registry["vit_b"](checkpoint=path)
    sam = sam.to(device)

    # mask_gen = SamPointMaskGenerator(
    #     sam,
    #     pred_iou_thresh=0.2,
    #     stability_score_thresh=0.2,
    #     min_mask_region_area=20 * 20
    # )

    mask_gen = SamPointMaskGenerator(
        sam,
        pred_iou_thresh=0,
        stability_score_thresh=0,
        min_mask_region_area=0
    )
    return mask_gen


def benchmark_cityscapes(n=500):
    save_folder = 'cityscapes_results'
    if not Path(save_folder).exists():
        Path(save_folder).mkdir(parents=True, exist_ok=True)

    dataset_base = PointCityscapes(cityscapes_root, split='val', mode='fine', point_type='single', target_type='instance')
    save_path = Path(save_folder) / f'cityscapes_base_{n}.pkl'
    print(f'evaluating cityscapes base {n} samples')
    eval_cityscapes_small(dataset_base, save_path=save_path, samples=n)

    dataset_rain = PointCityscapesRain(cityscapes_root, split='val', mode='fine', point_type='single', target_type='instance')
    save_path = Path(save_folder) / f'cityscapes_rain_{n}.pkl'
    print(f'evaluating cityscapes rain {n} samples')
    eval_cityscapes_small(dataset_rain, save_path=save_path, samples=n)

    dataset_fog = PointCityscapesFog(cityscapes_root, split='val', mode='fine', point_type='single', target_type='instance')
    save_path = Path(save_folder) / f'cityscapes_fog_{n}.pkl'
    print(f'evaluating cityscapes fog {n} samples')
    eval_cityscapes_small(dataset_fog, save_path=save_path, samples=n)


def test_cityscapes_instance():
    mask_gen = load_sam()
    dataset = PointCityscapes(cityscapes_root, split='val', mode='fine', point_type='single', target_type='instance')

    #  go over dataset
    for i in range(12, 14):
        image, target, points, labels = dataset[i]
        points = points[:, :]
        labels = labels[:, :]
        print(points.shape)
        # 'segmentation', 'segmentation_raw', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'
        out = mask_gen.generate(image, points, labels)
        # plt.figure(figsize=(8, 4))
        # plt.imshow(image)
        # plt.axis('off')
        # plt.show()
        plt.figure(figsize=(8, 4))
        plt.imshow(image)
        show_anns(out)
        plt.axis('off')
        plt.show()
        # show the raw masks
        n_masks = len(out)
        # create subplots for each mask
        fig, axs = plt.subplots(n_masks, 2, figsize=(8, 2 * n_masks))
        for m_idx, (ax, o) in enumerate(zip(axs, out)):
            pt_idx = o['point_idx'][0]
            m_pts = points[pt_idx]
            m_labels = labels[pt_idx]

            mask_raw = o['segmentation_raw'].detach().cpu()
            axs[m_idx][0].imshow(mask_raw)
            axs[m_idx][0].axis('off')

            axs[m_idx][1].imshow(image)
            # rescale points according to image size
            m_pts[:, 0] *= image.shape[1]
            m_pts[:, 1] *= image.shape[0]
            # draw points in mask
            axs[m_idx][0].scatter(m_pts[:, 0], m_pts[:, 1], s=3, c='red', marker='x')
            show_points(m_pts, m_labels, axs[m_idx][1])
            axs[m_idx][1].axis('off')
        plt.show()

        # if i > 2:
        #     break


def eval_cityscapes_small(dataset, save_path=None, samples=10):
    pred_threshs = torch.arange(0.1, 1, 0.05).tolist()

    # iou for multiple threshold values
    # TODO consider multi-class eval, implement soft ious version
    ious = MetricCollection({f'iou_{int(t * 100)}':
                                      BinaryJaccardIndex(threshold=t) for t in pred_threshs}).to(device)

    mask_gen = load_sam()

    im_results = {}
    if samples < 0:
        samples = len(dataset)
    for i in tqdm(range(samples)):
    # for i in [242, 243, 245]:
        image, gt_mask, points, labels = dataset[i]
        if len(points) == 0:
            continue
        gt_mask = torch.tensor(gt_mask).to(device)
        points = points[:, :]
        labels = labels[:, :]
        # 'segmentation', 'segmentation_raw', 'area', 'bbox', 'predicted_iou', 'point_coords', 'stability_score', 'crop_box'
        with torch.no_grad():
            output_annots = mask_gen.generate(image, points, labels)
        # denormalize points
        points = (points * (image.shape[1], image.shape[0])).round().astype(int)
        im_gts, im_preds = [], []
        for idx, out in enumerate(output_annots):
            pt_idx = out['point_idx'][0]
            instance_pt = points[pt_idx][0]
            instance_pred = out['segmentation_raw']
            instance_gt = gt_mask == gt_mask[instance_pt[1], instance_pt[0]]
            im_gts.append(instance_gt)
            im_preds.append(instance_pred)
        im_res = ious(torch.stack(im_preds), torch.stack(im_gts))
        for k, v in im_res.items():
            im_results.setdefault(k, []).append(v.item())
    final_res = ious.compute()
    print(final_res)
    #     save im_results
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(im_results, f)



if __name__ == '__main__':
    # so that we always sample same points
    np.random.seed(0)
    # test_cityscapes_single_point()
    # test_cityscapes_instance()
    # eval_citiscapes_small()
    benchmark_cityscapes()