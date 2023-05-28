import sys
import os
import requests
import subprocess

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image


sys.path.append('..')
import models_mae
from util.datasets_reconstruct import SingleClassImageFolder


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def wgetpp(url, path=''):
    # TODO add -nc, -r as params
    subprocess.run(["wget", "-nc", "-P", path, url])
    filepath = os.path.join(path, url.split('/')[-1])
    assert os.path.exists(filepath), "Something is wrong with the filename creation logic or url"
    return filepath


import collections


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.OrderedDict):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def download_models():
    # url = 'https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth'

    wgetpp('https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth', path='ckpts/')

    # chkpt_dir = wgetpp(url)
    # model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    # doesn't contain 'model_mask_token' and the whole decoder
    # cpt1 = torch.load('mae_pretrain_vit_large.pth', map_location='cpu')
    # cpt2 = torch.load('mae_visualize_vit_large.pth', map_location='cpu')
    #
    # keys1 = flatten(cpt1).keys()
    # keys2 = flatten(cpt2).keys()

    # model_mae_b = prepare_model('mae_pretrain_vit_base.pth', 'mae_vit_base_patch16')
    # model_mae_l = prepare_model('mae_pretrain_vit_large.pth', 'mae_vit_large_patch16')

if __name__ == '__main__':

    model = getattr(models_mae, 'mae_vit_base_patch16_seg')(img_size=384)
    # load model
    checkpoint = torch.load('ckpts/mae_visualize_vit_base.pth', map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)

    # download_models()

    #
    #
    # data = SingleClassImageFolder(root='/Users/panda/Technion/datasets/icon-datasets/DUTS/Train')
    # im, label = data[0]
    # plt.imshow(im)
    # plt.show()
    #
    # print()

    # from torchvision.datasets import CocoDetection
    #
    # coco = CocoDetection()
    #
    # blrs = [5e-6, 1e-5, 5e-5]
    # acc_iters = [2]
    # c = 0
    # start_c = 2
    # for accum_iter in acc_iters:
    #     for blr in blrs:
    #         if c < start_c:
    #             print('not running', c)
    #             c += 1
    #             continue
    #         print('running', c)
    #         c += 1
