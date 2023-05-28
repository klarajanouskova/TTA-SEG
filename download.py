import sys
import os
import requests
import subprocess
from pathlib import Path



class COCO:
    COCO_IMAGES_UNLABELED_2017 = "http://images.cocodataset.org/zips/unlabeled2017.zip"
    COCO_IMAGES_TRAIN_2017 = 'http://images.cocodataset.org/zips/train2017.zip'
    COCO_IMAGES_VAL_2017 = 'http://images.cocodataset.org/zips/val2017.zip'
    COCO_IMAGES_TEST_2017 = 'http://images.cocodataset.org/zips/test2017.zip'

    COCO_IMAGES_UNLABELED_2017_INFO = 'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip'
    COCO_IMAGES_TEST_2017_INFO = 'http://images.cocodataset.org/annotations/image_info_test2017.zip'


class KITTI:
    pass

class Cityscapes:
    # 1 -> gtFine_trainvaltest.zip (241MB)
    # 2 -> gtCoarse.zip (1.3GB)
    # 3 -> leftImg8bit_trainvaltest.zip (11GB)
    # 4 -> leftImg8bit_trainextra.zip (44GB)
    # 8 -> camera_trainvaltest.zip (2MB)
    # 9 -> camera_trainextra.zip (8MB)
    # 10 -> vehicle_trainvaltest.zip (2MB)
    # 11 -> vehicle_trainextra.zip (7MB)
    # 12 -> leftImg8bit_demoVideo.zip (6.6GB)
    # 28 -> gtBbox_cityPersons_trainval.zip (2.2MB)
    subset2id = {
    'TRAINVAL_IMS_FINE_GT': 1,
    'COARSE_GT': 2,
    'TRAINVAL_IMS': 3,
    'TRAINVAL_IMS_EXTRA': 4,
    'FOG_TRAINVAL_IMS': 31,
    'RAIN_TRAINVAL_IMS': 33
    }

class ACDC:
    GT_SEM_SEG =' https://acdc.vision.ee.ethz.ch/gt_trainval.zip'
    IMGS = 'https://acdc.vision.ee.ethz.ch/rgb_anon_trainvaltest.zip'


def wgetpp(url, path=''):
    # TODO add -nc, -r as params
    Path(path).mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-nc", "-P", path, url])
    filepath = os.path.join(path, url.split('/')[-1])
    assert os.path.exists(filepath), "Something is wrong with the filename creation logic or url"
    return filepath


def download_coco_images():
    coco = COCO()
    wgetpp(coco.COCO_IMAGES_UNLABELED_2017, '/mnt/walkure_pub/klara/coco')

def download_bdd100k():
    pass
    # TODO  there is probably no public url link, you need to log in and download it manually
    # do sth similar to cityscapes...


def download_cityscapes():
    import shlex

    def load_cityscapes_pswd():
        with open('../cityscapes_pswd.txt', 'r') as f:
            return f.read().strip()

    def download_subset(subset):
        path = '/datagrid/TextSpotter/klara/datasets/cityscapes'
        Path(path).mkdir(parents=True, exist_ok=True)
        pswd = load_cityscapes_pswd()
        args = shlex.split(
            f"wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=klara&password={pswd}&submit=Login' https://www.cityscapes-dataset.com/login/")
        subprocess.run(args)
        args = shlex.split(
            f"wget --load-cookies=cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID={cityscapes.subset2id[subset]} -P {path}")
        subprocess.run(args)

    cityscapes = Cityscapes()
    for subset in ['TRAINVAL_IMS_FINE_GT', 'TRAINVAL_IMS', 'FOG_TRAINVAL_IMS', 'RAIN_TRAINVAL_IMS']:
        download_subset(subset)


def download_acdc():
    acdc = ACDC()
    wgetpp(acdc.GT_SEM_SEG, '/mnt/walkure_pub/klara/acdc')
    wgetpp(acdc.IMGS, '/mnt/walkure_pub/klara/acdc')

if __name__ == '__main__':
    download_cityscapes()