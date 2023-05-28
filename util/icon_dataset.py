#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import argparse
from pathlib import Path


from util.datasets_reconstruct import split_dataset

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask=None, body=None, detail=None):
        image = (image - self.mean) / self.std
        if mask is None:
            return image
        return image, mask / 255


class RandomCrop(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        if mask is None:
            return image[p0:p1, p2:p3, :]
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        if np.random.randint(2) == 0:
            if mask is None:
                return image[:, ::-1, :].copy()
            return image[:, ::-1, :].copy(), mask[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask=None, body=None, detail=None):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        if mask is None:
            return image
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # body = cv2.resize(body, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        # detail = cv2.resize(detail, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask


class RandomRotate(object):
    def rotate(self, x, random_angle, mode='image'):

        if mode == 'image':
            H, W, _ = x.shape
        else:
            H, W = x.shape

        random_angle %= 360
        image_change = cv2.getRotationMatrix2D((W / 2, H / 2), random_angle, 1)
        image_rotated = cv2.warpAffine(x, image_change, (W, H))

        angle_crop = random_angle % 180
        if random_angle > 90:
            angle_crop = 180 - angle_crop
        theta = angle_crop * np.pi / 180
        hw_ratio = float(H) / float(W)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
        r = hw_ratio if H > W else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        w_crop = int(crop_mult * W)
        h_crop = int(crop_mult * H)
        x0 = int((W - w_crop) / 2)
        y0 = int((H - h_crop) / 2)
        crop_image = lambda img, x0, y0, W, H: img[y0:y0 + h_crop, x0:x0 + w_crop]
        output = crop_image(image_rotated, x0, y0, w_crop, h_crop)

        return output

    def __call__(self, image, mask=None, body=None, detail=None):

        do_seed = np.random.randint(0, 3)
        if do_seed != 2:
            if mask is None:
                return image
            return image, mask

        random_angle = np.random.randint(-10, 10)
        image = self.rotate(image, random_angle, 'image')

        if mask is None:
            return image
        mask = self.rotate(mask, random_angle, 'mask')

        return image, mask


class ColorEnhance(object):
    def __init__(self):

        # A:0.5~1.5, G: 5-15
        self.A = np.random.randint(7, 13, 1)[0] / 10
        self.G = np.random.randint(7, 13, 1)[0]

    def __call__(self, image, mask=None, body=None, detail=None):

        do_seed = np.random.randint(0, 3)
        if do_seed > 1:  # 1: # 1/3
            H, W, _ = image.shape
            dark_matrix = np.zeros([H, W, _], image.dtype)
            image = cv2.addWeighted(image, self.A, dark_matrix, 1 - self.A, self.G)
        else:
            pass

        if mask is None:
            return image
        return image, mask


class GaussNoise(object):
    def __init__(self):
        self.Mean = 0
        self.Var = 0.001

    def __call__(self, image, mask=None, body=None, detail=None):
        H, W, _ = image.shape
        do_seed = np.random.randint(0, 3)

        if do_seed == 0:  # 1: # 1/3
            factor = np.random.randint(0, 10)
            noise = np.random.normal(self.Mean, self.Var ** 0.5, image.shape) * factor
            noise = noise.astype(image.dtype)
            image = cv2.add(image, noise)
        else:
            pass

        if mask is None:
            return image
        return image, mask


class GaussNoiseCorr(object):
    def __init__(self, scale=0.18):
        self.Mean = 0
        self.scale = scale

    def __call__(self, image, mask=None, body=None, detail=None):
        H, W, _ = image.shape

        factor = np.random.randint(0, 10)
        noise = np.random.normal(self.Mean, self.scale, image.shape) * factor
        noise = noise.astype(image.dtype)
        image = cv2.add(image, noise)


        if mask is None:
            return image
        return image, mask


class ToTensor(object):
    def __call__(self, image, mask=None, body=None, detail=None):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if mask is None:
            return image
        mask = torch.from_numpy(mask)
        return image, mask

    ########################### Config File ###########################


class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([102.94, 118.90, 124.55])
        self.std = np.array([57.50, 55.97, 56.77])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, root, transform=None, train=True, size=224):
        self.image_folder = os.path.join(root, 'Image')
        self.mask_folder = os.path.join(root, 'GT')
        # TODO sort these to ensure same order
        # currently images are filtered, not GT, so we are loading by imags
        self.samples = [Path(x).stem for x in os.listdir(self.image_folder) if Path(x.lower()).suffix in
                                  ['.jpg', '.png', '.jpeg']]

        self.sample2idx = {x: i for i, x in enumerate(self.samples)}
        self.idx2sample = {i: x for i, x in enumerate(self.samples)}

        self.transform = transform
        self.train = train
        self.size = size

    def __getitem__(self, idx):
        name = self.samples[idx]
        try:
            image = cv2.imread(os.path.join(self.image_folder, name + '.jpg'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print("######" + str(name))

        try:
            mask = cv2.imread(os.path.join(self.mask_folder, name + '.png'), 0).astype(np.float32)
        except:
            print("#######" + str(name))
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, name
        # if not train
        # else:
        #     if self.transform:
        #         image = self.transform(image)
        #     return image, name

    def __len__(self):
        return len(self.samples)

    def collate(self, batch):

        image, mask, name = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2).float()
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        return image, mask, name


class Transformation():
    def __init__(self, mean=np.array([[[102.94, 118.90, 124.55]]]), std=np.array([[[57.50, 55.97, 56.77]]]),
                 size=224, train=True, corrupt_scale=None):
        # before ToTensor: imagenet?? 0,255 stats
        self.normalize = Normalize(mean=mean, std=std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize = Resize(size, size)
        # self.randomrotate = RandomRotate()
        # self.colorenhance = ColorEnhance()
        # self.gaussnoise = GaussNoise()
        self.corrupt_scale = corrupt_scale
        if corrupt_scale is not None:
            self.corrupt = GaussNoiseCorr(scale=corrupt_scale)
        # imagenet values
        # mean = [0.485, 0.456, 0.406],
        # std = [0.229, 0.224, 0.225]
        self.totensor = ToTensor()
        self.train = train

    def __call__(self, image, mask=None):
        assert not (self.train and mask is None), 'Train phase must have mask'
        if self.corrupt_scale:
            image = self.corrupt(image)
        if mask is not None:
            image, mask = self.normalize(image, mask)
            if self.train:
                # image, mask = self.randomcrop(image, mask)
                image, mask = self.randomflip(image, mask)
                # image, mask = self.randomrotate(image, mask)
                # image, mask = self.colorenhance(image, mask)
                # image, mask = self.gaussnoise(image, mask)
            image, mask = self.resize(image, mask)
            return image, mask
        else:
            if self.train:
                image = self.normalize(image)
                image = self.randomflip(image)
            else:
                image = self.normalize(image)
                image = self.resize(image)
            return image





def get_icon(args, global_rank, num_tasks):
    path = os.path.join('icon-datasets', args.dataset, 'Train')

    dataset = Data(root=os.path.join(args.data_path, path), size=args.input_size)

    transform_train = Transformation(train=True, size=args.input_size)
    transform_val = Transformation(train=False, size=args.input_size)

    dataset_train, dataset_val = split_dataset(dataset, args, transform_train, transform_val, masks=True)
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))

    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, num_workers=args.num_workers,
                                  pin_memory=True, drop_last=True, collate_fn=dataset.collate)
    dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, drop_last=False, collate_fn=dataset.collate)
    return dataloader_train, dataloader_val


def get_trainval_dataloaders(args, global_rank, num_tasks):
    """
    Splits the dataset into train and validation sets and returns the dataloaders for both.
    """
    if args.dataset == 'DUTS':
        return get_icon(args, global_rank, num_tasks)

def get_test_dataloader(args, dataset_name):
    """
    Returns the dataloader for the test set.
    """
    path = os.path.join('icon-datasets', dataset_name, 'Test')

    transform = Transformation(train=False, size=args.input_size)

    dataset = Data(root=os.path.join(args.data_path, path), size=args.input_size, transform=transform)

    subset = None
    # subset = torch.utils.data.Subset(dataset, range(10))

    sampler_test = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(subset if subset is not None else dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                                pin_memory=True, drop_last=False, collate_fn=dataset.collate)
    return dataloader


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    # test segmentation ViT
    transform = Transformation()
    data = Data(root='../../datasets/icon-datasets/DUTS/Train', transform=transform)
    (img, mask), name = data[0]
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(mask.squeeze())
    plt.show()
    # remember to set up collate function properly!
    loader = DataLoader(data, collate_fn=data.collate)