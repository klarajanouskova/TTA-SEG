# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

import torch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class SingleClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, class_folder='Img', transform=None):
        """
        Args:
            root (string): Root directory path.
            class_folder (string): Name of the folder with the class images .
            transform (callable, optional): A function/transform that  takes in an PIL image
        """
        self.cls = class_folder
        super(SingleClassImageFolder, self).__init__(root, transform)

    def find_classes(self, directory: str):
        classes = [self.cls]
        class_to_idx = {self.cls: 0}
        return classes, class_to_idx


def get_transform_random_crop(args):
    return transforms.Compose([
        # transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic ..ORIG
        transforms.RandomResizedCrop(args.input_size, scale=(0.7, 1.0), interpolation=3),  # 3 is bicubic

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # keep imagenet stats for normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def get_transform_five_crop(args):
    return transforms.Compose([
        transforms.ToTensor(),
        # keep imagenet stats for normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ResizeIfNeeded(args.input_size, interpolation=3),
        transforms.FiveCrop(args.input_size)]
    )


def datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks):
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            # divide by 5 because of fivecrop aug
            batch_size=args.batch_size // 5,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    return data_loader_train, data_loader_val


def get_pascal(args, global_rank, num_tasks):
    path = 'VOC/VOCdevkit/VOC2012'
    im_folder = 'JPEGImages'

    dataset = SingleClassImageFolder(root=os.path.join(args.data_path, path), class_folder=im_folder)

    transform_train = get_transform_random_crop(args)

    transform_val = get_transform_five_crop(args)

    dataset_train, dataset_val = split_dataset(dataset, args, transform_train, transform_val)

    return datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks)


def get_DUTS(args, global_rank, num_tasks):
    path = 'icon-datasets/DUTS/Train'
    im_folder = 'Image'

    dataset = SingleClassImageFolder(root=os.path.join(args.data_path, path), class_folder=im_folder)

    transform_train = get_transform_random_crop(args)

    transform_val = get_transform_five_crop(args)

    dataset_train, dataset_val = split_dataset(dataset, args, transform_train, transform_val)

    return datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks)


def get_BDD10K(args, global_rank, num_tasks):
    path = 'bdd100k/images/10k/'
    im_folder = 'train'

    dataset = SingleClassImageFolder(root=os.path.join(args.data_path, path), class_folder=im_folder)

    transform_train = get_transform_random_crop(args)

    transform_val = get_transform_five_crop(args)

    dataset_train, dataset_val = split_dataset(dataset, args, transform_train, transform_val)

    return datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks)


def get_BDD100K(args, global_rank, num_tasks):
    path = 'bdd100k/images/100k/'
    im_folder = 'train'

    dataset = SingleClassImageFolder(root=os.path.join(args.data_path, path), class_folder=im_folder)

    transform_train = get_transform_random_crop(args)

    transform_val = get_transform_five_crop(args)

    dataset_train, dataset_val = split_dataset(dataset, args, transform_train, transform_val)

    return datasets2dataloaders(dataset_train, dataset_val, args, global_rank, num_tasks)


def get_dataloaders(args, global_rank, num_tasks):
    if args.dataset == 'DUTS':
        return get_DUTS(args, global_rank, num_tasks)
    if args.dataset == 'pascal':
        return get_pascal(args, global_rank, num_tasks)
    if args.dataset == 'BDD10k':
        return get_BDD10K(args, global_rank, num_tasks)
    if args.dataset == 'BDD100k':
        return get_BDD100K(args, global_rank, num_tasks)


def split_dataset(dataset, args, transform_train, transform_val, masks=False):
    subset_class = DatasetFromSubset if not masks else DatasetFromSubsetSeg
    if args.valid_part > 0:
        n_data = len(dataset)
        if args.valid_part >= 1:
            n_valid = int(args.valid_part)
        else:
            n_valid = int(args.valid_part * n_data)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_data - n_valid, n_valid])
        dataset_val = subset_class(dataset_val, transform=transform_val)
    else:
        dataset_train, dataset_val = dataset, None

    dataset_train = subset_class(dataset_train, transform=transform_train)
    return dataset_train, dataset_val


class ResizeIfNeeded(transforms.Resize):

    def forward(self, img):
        h, w = img.size()[1:]
        if h >= self.size and w >= self.size:
            return img
        else:
            # If size is an int, smaller edge of the image will be matched to this number
            return transforms.functional.resize(img, self.size, self.interpolation, self.max_size, self.antialias)


class DatasetFromSubset(torch.utils.data.Dataset):
    """
    Subset dataset wrapper that adds transform to the subset dataset
    """
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


class DatasetFromSubsetSeg(DatasetFromSubset):
    """
    Takes common transformation for images and masks into account
    """
    def __getitem__(self, index):
        data = self.subset[index]
        if len(data) == 3:
            x, y, z = data
            if self.transform:
                x, y = self.transform(x, y)
            return x, y, z
        else:
            x, z = data
            if self.transform:
                x = self.transform(x)
        return x, z

    def __len__(self):
        return len(self.subset)



