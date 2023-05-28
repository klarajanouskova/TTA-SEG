import os

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import disable_beta_transforms_warning
disable_beta_transforms_warning()
import torchvision.transforms.v2 as tfms
import torchvision.transforms.v2.functional as F
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import utils
from torchvision import datapoints as dp
import torch
from torchvision.transforms.v2 import functional as F, Transform
from typing import Any, Dict


from util.voc_dataset_seg import VOCSegmentationSubFgBg, HBBoxTransform, CATS_A, CATS_B

# implemented datasets, keep the name lowercase
DATASETS = ['pascal', 'duts']

# Exactly the same interface as V1:
trans = tfms.Compose([
    tfms.ColorJitter(contrast=0.5),
    tfms.RandomRotation(30),
    tfms.CenterCrop(480),
])


def get_pascal(args, split=None):
    def get_pascal_train():
        return VOCSegmentationSubFgBg(root=pasacl_root,
                               sub=args.data_cls_sub,
                               transform=get_train_transform(args),
                               bbox_transform=bbox_trans,
                               image_set="train")
    def get_pascal_val():
        return VOCSegmentationSubFgBg(root=pasacl_root,
                                         sub=args.data_cls_sub,
                                         transform=get_test_transform(args),
                                         bbox_transform=bbox_trans,
                                            image_set="val")
    assert split in ['train', 'val', None]
    bbox_trans = HBBoxTransform(range=(0.4, 0.4))

    pasacl_root = os.path.join(args.data_path, 'VOC')

    if split is None:
        return get_pascal_train(), get_pascal_val()
    elif split == 'train':
        return get_pascal_train()
    else:
        return get_pascal_val()


def get_dataset(args):
    assert args.dataset in DATASETS, f"Dataset {args.dataset} not implemented"
    if args.dataset == 'pascal':
        return get_pascal(args)


def get_dataloaders(args, global_rank, num_tasks):
    dataset_train, dataset_val = get_dataset(args)
    return datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks)


def get_train_transform(args):
    if args.preserve_aspect:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            tfms.RandomHorizontalFlip(),
            # so that the model has seen the padding during training
            PadToSquare(fill=0),
            tfms.Resize(size=(args.input_size, args.input_size), antialias=True),
            # do not crop or distort too much
            tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.6, 1.), ratio=(0.9, 1.1),
                                   antialias=True)
        ])
    else:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            tfms.RandomHorizontalFlip(),
            # so that the model has seen the padding during training
            # PadToSquare(fill=0),
            # tfms.Resize(size=(args.input_size, args.input_size), antialias=True),
            # do not crop or distort too much
            tfms.RandomResizedCrop(size=(args.input_size, args.input_size), scale=(0.6, 1.), ratio=(0.9, 1.1), antialias=True)
        ])
    return transform


def get_train_transform_experimental(args):
    if args.preserve_aspect:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            tfms.AutoAugment(policy=tfms.AutoAugmentPolicy.IMAGENET),
            PadToSquare(fill=0),
            tfms.Resize(size=(args.input_size, args.input_size), antialias=True)
        ])
    else:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            tfms.AutoAugment(policy=tfms.AutoAugmentPolicy.IMAGENET),
            tfms.Resize(size=(args.input_size, args.input_size), antialias=True)
        ])
    return transform


def get_test_transform(args):
    if args.preserve_aspect:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            PadToSquare(fill=0),
            tfms.Resize(size=(args.input_size, args.input_size), antialias=True)
        ])
    else:
        transform = tfms.Compose([
            tfms.ToImageTensor(),
            tfms.ConvertImageDtype(dtype=torch.float32),
            tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            tfms.Resize(size=(args.input_size, args.input_size), antialias=True)
        ])
    return transform


class PadToSquare(Transform):
    def __init__(self, fill=0, padding_mode='constant'):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        h, w = inpt.shape[-2:]
        if h == w:
            return inpt
        dim_diff = abs(h - w)
        # (upper / left) padding and (lower / right) padding
        pad_size = dim_diff // 4
        # Determine padding
        # left, top, right and bottom
        pad = (pad_size, 0, pad_size, 0) if h >= w else (0, pad_size, 0, pad_size)
        return F.pad(inpt, padding=pad, fill=self.fill, padding_mode=self.padding_mode)


def datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = get_train_dataloader(dataset_train, args, global_rank, num_tasks)

    if dataset_val:
        data_loader_val = get_test_dataloader(dataset_val, args)
    else:
        data_loader_val = None

    return data_loader_train, data_loader_val


def get_train_dataloader(dataset, args, global_rank, num_tasks):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
    )
    return data_loader_train


def get_test_dataloader(dataset, args):
    sampler_val = torch.utils.data.SequentialSampler(dataset)
    data_loader_test = torch.utils.data.DataLoader(
        dataset,
        sampler = sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return data_loader_test


def test_random():
    h, w, c = 480, 480, 3
    img = (torch.rand(c, h, w) * 255).to(torch.uint8)

    masks = torch.zeros(2, h, w)
    masks[0, 100:200, 100:200] = 1
    masks[1, 200:300, 200:300] = 1
    masks = masks.bool()
    img = dp.Image(img)
    masks = dp.Mask(masks)

    trans = tfms.Compose([tfms.ToImageTensor(), tfms.ConvertImageDtype(dtype=torch.float32),
                          tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    # revert the normalization back to [0, 1] and then [0, 255] for visualization
    denorm = tfms.Compose([tfms.Normalize(mean=[-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
                                          std=[1 / s for s in IMAGENET_DEFAULT_STD]),
                           tfms.ConvertImageDtype(dtype=torch.uint8)])

    vis_im = utils.draw_segmentation_masks(F.to_image_tensor(img), masks=masks)
    t_img, t_masks = trans(img, masks)
    t_vis_im = utils.draw_segmentation_masks(F.convert_dtype(t_img, dtype=torch.uint8), masks=t_masks)
    d_img, d_masks = denorm(t_img, t_masks)
    d_vis_im = utils.draw_segmentation_masks(d_img, masks=d_masks)

    #     show images with plt
    plt.imshow(F.to_pil_image(vis_im))
    plt.title("original image")
    plt.show()
    plt.imshow(F.to_pil_image(t_vis_im))
    plt.title("transformed image")
    plt.show()
    plt.imshow(F.to_pil_image(d_vis_im))
    plt.title("denormalized image")
    plt.show()


def test_image(img_path="../otter.jpeg"):
    # h, w, c = 480, 480, 3
    img = Image.open("../animals/otter.jpeg")
    w, h, c = *img.size, 3
    # img = (torch.rand(c, h, w) * 255).to(torch.uint8)

    masks = torch.zeros(2, h, w)
    masks[0, 100:200, 100:200] = 1
    masks[1, 200:300, 200:300] = 1
    masks = masks.bool()
    img = dp.Image(img)
    masks = dp.Mask(masks)

    trans = tfms.Compose([tfms.ToImageTensor(), tfms.ConvertImageDtype(dtype=torch.float32), tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])

    # revert the normalization back to [0, 1] and then [0, 255] for visualization
    denorm = tfms.Compose([tfms.Normalize(mean=[-m / s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
                                        std=[1 / s for s in IMAGENET_DEFAULT_STD]),
                           tfms.ConvertImageDtype(dtype=torch.uint8)])


    vis_im = utils.draw_segmentation_masks(F.to_image_tensor(img), masks=masks)
    t_img, t_masks = trans(img, masks)
    t_vis_im = utils.draw_segmentation_masks(F.convert_dtype(t_img, dtype=torch.uint8), masks=t_masks)
    d_img, d_masks = denorm(t_img, t_masks)
    d_vis_im = utils.draw_segmentation_masks(d_img, masks=d_masks)

    #     show images with plt
    plt.imshow(F.to_pil_image(vis_im))
    plt.title("original image")
    plt.show()
    plt.imshow(F.to_pil_image(t_vis_im))
    plt.title("transformed image")
    plt.show()
    plt.imshow(F.to_pil_image(d_vis_im))
    plt.title("denormalized image")
    plt.show()


def test_pascal():
    bbox_trans = HBBoxTransform(range=(0.15, 0.15))
    transform = tfms.Compose([tfms.ToImageTensor(),
                              tfms.ConvertImageDtype(dtype=torch.float32),
                                tfms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
                              tfms.Resize((384, 384)),
                              tfms.RandomRotation(30)])
    dataset = VOCSegmentationSubFgBg(root="/Users/panda/datasets/voc",
                                     sub="all",
                                     transform=None,
                                     bbox_transform=bbox_trans)

    im, mask, cls_targets, name = dataset[5]

    t_im, t_mask = transform(im, mask)

    vis_im = utils.draw_segmentation_masks(im, masks=mask)
    t_vis_im = utils.draw_segmentation_masks(F.convert_dtype(t_im, dtype=torch.uint8), masks=t_mask)
    plt.imshow(F.to_pil_image(vis_im))
    plt.title("original image")
    plt.show()
    plt.imshow((F.to_pil_image(t_vis_im)))
    plt.title("transformed image")
    plt.show()


def test_pascal2():
    from arg_composition import get_segmentation_args
    args = get_segmentation_args().parse_args()
    args.num_workers = 0

    dl_train, dl_val = get_dataloaders(args, 0, 1)
    for batch in dl_val:
        im, mask, cls_targets, name = batch
        vis_im = utils.draw_segmentation_masks(F.convert_dtype(im[0], dtype=torch.uint8),
                                               masks=F.convert_dtype(mask[0], dtype=torch.bool))
        plt.imshow(F.to_pil_image(vis_im))
        plt.title("original image")
        plt.show()
        break


if __name__ == '__main__':
    # test_image()
    # test_random()
    test_pascal()
    # test_pascal2()