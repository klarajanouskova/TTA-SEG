from typing import Any, Callable, Dict, Optional, Tuple, List
import os
import warnings

import numpy as np
from PIL import Image

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.voc import _VOCBase, verify_str_arg, download_and_extract_archive, DATASET_YEAR_DICT
# TODO rewrite using dps
from torchvision import datapoints as dp


ROOT = "/Users/panda/datasets/voc"
CATS = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
         'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
         'train', 'tvmonitor', 'ignore']
CLASSES = [x for x in range(len(CATS) - 1)] + [255]
COLORMAP = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
            (224, 224, 192)]
CAT2IDX = {cat: idx for idx, cat in enumerate(CATS)}
IDX2CAT = {idx: cat for idx, cat in enumerate(CATS)}
CLS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX2CLS = {idx: cls for idx, cls in enumerate(CLASSES)}
CAT2CLS = {cat: CLASSES[idx] for idx, cat in enumerate(CATS)}
CLS2CAT = {cls: cat for cat, cls in CAT2CLS.items()}
CLS2COLOR = {cls: COLORMAP[idx] for idx, cls in enumerate(CLASSES)}
CAT2COLOR = {cat: COLORMAP[idx] for idx, cat in enumerate(CATS)}


# categories split into two sets, the 1st, 3nd, 5th, ... most common classes are in A, 2nd, 4th, ... in B
CATS_A = ['person', 'car', 'cat', 'bottle', 'tvmonitor', 'train', 'pottedplant', 'boat', 'horse', 'sheep']
CATS_B = ['dog', 'chair', 'bird', 'aeroplane', 'bicycle', 'diningtable', 'motorbike', 'sofa', 'bus', 'cow']


class VOCClassification(_VOCBase):
    """
    Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Classification Dataset.

        Args:
            root (string): Root directory of the VOC Dataset.
            year (string, optional): The dataset year, supports years ``"2007"`` to ``"2012"``.
            image_set (string, optional): Select the image_set to use, ``"train"``, ``"trainval"`` or ``"val"``. If
                ``year=="2007"``, can also be ``"test"``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
            transforms (callable, optional): A function/transform that takes input sample and its target as entry
                and returns a transformed version.
        """

    _SPLITS_DIR = "Main"
    _TARGET_FILE_EXT = ".txt"

    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "train",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        VisionDataset.__init__(self, root, transforms, transform, target_transform)
        if year == "2007-test":
            if image_set == "test":
                warnings.warn(
                    "Acessing the test image set of the year 2007 with year='2007-test' is deprecated. "
                    "Please use the combination year='2007' and image_set='test' instead."
                )
                year = "2007"
            else:
                raise ValueError(
                    "In the test image set of the year 2007 only image_set='test' is allowed. "
                    "For all other image sets use year='2007' instead."
                )
        self.year = year

        valid_image_sets = ["train", "trainval", "val"]
        if year == "2007":
            valid_image_sets.append("test")
        self.image_set = verify_str_arg(image_set, "image_set", valid_image_sets)

        key = "2007-test" if year == "2007" and image_set == "test" else year
        dataset_year_dict = DATASET_YEAR_DICT[key]

        self.url = dataset_year_dict["url"]
        self.filename = dataset_year_dict["filename"]
        self.md5 = dataset_year_dict["md5"]

        if download:
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)

        base_dir = dataset_year_dict["base_dir"]
        self.voc_root = os.path.join(self.root, base_dir)
        if not os.path.isdir(self.voc_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        self.image_dir = os.path.join(self.voc_root, "JPEGImages")

        # global index is the official one, local is for this dataset implementation
        self.loc2globcls = {idx: cls for idx, cls in enumerate(CLASSES[1:-1])}
        self.glob2loccls = {cls: idx for idx, cls in enumerate(CLASSES[1:-1])}

        if not os.path.exists(self.cls_file):
            splits_dir = os.path.join(self.voc_root, "ImageSets", self._SPLITS_DIR)
            print(f'Classification GT {self.cls_file} does not exist, it will be created now.')
            self.generate_cls_file(splits_dir, self.image_set, self.glob2loccls, self.cls_file)
            print('Classification GT has been created successfully!')
        self.images, self.targets = self.load_dataset()

        assert len(self.images) == len(self.targets)

    @property
    def cls_file(self):
        return os.path.join(self.voc_root, "ImageSets", self._SPLITS_DIR, f'cls_{self.image_set}.npy')

    def generate_cls_file(self, splits_dir, image_set, glob2loc, cls_file_path):
        # TODO rewrite without params
        imnames = []
        target_dict = {}

        for cat in CATS[1:-1]:
            loc_cls = glob2loc[CAT2CLS[cat]]
            txt_path = os.path.join(splits_dir, f"{cat}_{image_set}.txt")

            with open(txt_path) as f:
                split_lines = [line.split() for line in f.readlines()]
                cat_imnames = [imname for imname, cls in split_lines if int(cls) > 0]
                for im in cat_imnames:
                    im_label = target_dict[im] if im in target_dict.keys() else np.zeros(len(CATS) - 2)
                    im_label[loc_cls] = 1
                    target_dict[im] = im_label
                imnames += cat_imnames

        im_set = sorted(set(imnames))
        targets = np.array([target_dict[im] for im in im_set])
        np.save(cls_file_path, {'array': targets, 'names': np.array(im_set), 'dict': target_dict})

    def get_cls_distribution(self):
        cls_array = np.load(self.cls_file, allow_pickle=True).item()['array']
        cls_im_counts = cls_array.sum(axis=0)
        sorted_idxs = sorted([x for x in range(len(cls_im_counts))], key=lambda x: cls_im_counts[x], reverse=True)
        for idx in sorted_idxs:
            print(f'{CATS[idx + 1]}: {cls_im_counts[idx]}')

        print([CATS[idx + 1] for idx in sorted_idxs[::2]])
        print([CATS[idx + 1] for idx in sorted_idxs[1::2]])
        # for cls, count in zip(CATS[1:-1], cls_im_counts):
        #     print(f'{cls}: {count}')

    def load_dataset(self):
        data = np.load(self.cls_file, allow_pickle=True).item()
        return data['names'], data['array']

    def __getitem__(self, item):
        im_path = os.path.join(self.image_dir, self.images[item] + ".jpg")
        im = Image.open(im_path)
        target = self.targets[item]
        if self.transform:
            im = self.transform(im)
        if self.target_transform:
            target = self.target_transform(target)
        meta = {'image_dir': self.image_dir, 'name': self.images[item], 'clsfile': self.cls_file}
        return im, target, meta

    def __len__(self):
        return len(self.images)


class VOCClassificationSub(VOCClassification):
    def __init__(
            self,
            root: str,
            year: str = "2012",
            image_set: str = "trainval",
            sub: str = "dog&cat",
            download: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None,
    ):
        super(VOCClassificationSub, self).__init__(root, year, image_set, download, transform, target_transform, transforms)
        self.sub, self.cats = sort_and_verify_sub(sub)
        # used ie. for indexing of the general class file
        self.rel_loc_idxs = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        self.sub2loccls = {idx: loc_idx for idx, loc_idx in enumerate(self.rel_loc_idxs)}
        self.loc2subcls = {loc: sub for sub, loc in self.sub2loccls.items()}
        self.sub2globcls = {sub: self.loc2globcls[loc] for sub, loc in enumerate(self.rel_loc_idxs)}
        self.glob2subcls = {glob: sub for sub, glob in self.sub2globcls.items()}
        self.images, self.targets = self.generate_subdataset()

    def generate_subdataset(self):
        # load the annotation for the whole dataset
        data = np.load(self.cls_file, allow_pickle=True).item()
        names_cls, array_cls = data['names'], data['array']

        # take just the relevant subset
        loc_clses = [self.glob2loccls[CAT2CLS[cat]] for cat in self.cats]
        sub_idxs = np.unique(np.where(array_cls[:, loc_clses])[0])
        im_names = names_cls[sub_idxs]
        cls_targets = array_cls[sub_idxs][:, loc_clses]
        return im_names, cls_targets


def sort_and_verify_sub(sub):
    if sub == 'all':
        sub = '&'.join(CATS[1:-1])
    if sub == 'A':
        sub = '&'.join(CATS_A)
    if sub == 'B':
        sub = '&'.join(CATS_B)
    verified_clses = [verify_str_arg(cls, "class in sub", CATS[1:-1]) for cls in sub.split("&")]
    sorted_cats = sorted(verified_clses)
    assert len(sorted_cats) > 1, "Only datasets with more than 1 class are supported at the moment"
    return '&'.join(sorted_cats), sorted_cats


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # d = VOCSegmentation(root=ROOT)
    # dataset_trainval = VOCClassification(root=ROOT, image_set='trainval')
    # dataset_train = VOCClassification(root=ROOT, image_set='train')
    dataset_val = VOCClassification(root=ROOT, image_set='val')
