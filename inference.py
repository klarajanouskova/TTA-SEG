import os
import torch
import numpy as np
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import CenterCrop, ToTensor

import models_mae

sys.path.append('..')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * IMAGENET_STD + IMAGENET_MEAN) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def infer_folder(model, img_folder, save_folder='visualization', n=30, seed=0, ratio=0.75):
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = CenterCrop(224)
    Path(save_folder).mkdir(exist_ok=True, parents=True)
    img_names = [x for x in os.listdir(img_folder) if x.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    for img_name in tqdm(img_names[:min(n, len(img_names))]):
        img_path = os.path.join(img_folder, img_name)
        save_path = os.path.join(save_folder, img_name)

        img = Image.open(img_path)
        # img = img.resize((224, 224))
        # img = np.array(img) / 255.

        img = transform(img)
        img = np.array(img) / 255.

        assert img.shape == (224, 224, 3)

        # normalize by ImageNet mean and std
        img = img - IMAGENET_MEAN
        img = img / IMAGENET_STD

        orig, mask, out = infer_image(img, model, ratio)
        visualize_image(orig, mask, out, save_path=save_path)


def infer_image(img, model, ratio=0.75):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run TTA
    loss, y, mask = model(x.float(), mask_ratio=ratio)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # reshape to visualizable form
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    x = torch.einsum('nchw->nhwc', x)

    return x, mask, y


def visualize_image(orig, mask, out, show=False, save_path=None):
    # masked image
    im_masked = orig * (1 - mask)

    # TTA reconstruction pasted with visible patches
    im_paste = orig * (1 - mask) + out * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(orig[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(out[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    if show:
        plt.show()

if __name__ == '__main__':
    size = 'base'
    # ckpt_dir = f'mae_visualize_vit_{size}.pth'
    # mae_model = prepare_model(ckpt_dir, f'mae_vit_{size}_patch16')

    # ckpt_dir = f'ckpts/mae_visualize_vit_{size}.pth'
    # ckpt_dir2 = f'test/checkpoint-49.pth'
    # mae_model = prepare_model(ckpt_dir, f'mae_vit_{size}_patch16')

    dut_omron = '/Users/panda/Technion/datasets/icon-datasets/DUTS/Test/Image'
    # infer_folder(mae_model, dut_omron, save_folder=f'visualization/ft_test_49/DUTS')

    # pascal_voc = '/Users/panda/Technion/datasets/VOC/VOCdevkit/VOC2012/JPEGImages'
    # infer_folder(mae_model, pascal_voc, save_folder=f'visualization/mae-{size}-gan/Pacal-VOC')

    # # test BDD model on DUTS
    # ckpt_path_orig = '/Users/panda/Technion/TTA/ckpts/mae_visualize_vit_base.pth'
    # mae_model_orig = prepare_model(ckpt_path_orig, f'mae_vit_{size}_patch16')
    # infer_folder(mae_model_orig, dut_omron, save_folder=f'visualization/DUTS/orig_0.75', ratio=0.75)
    # ckpt_path_ft = '/Users/panda/Technion/TTA/ckpts/best2.pth'
    # mae_model_ft = prepare_model(ckpt_path_ft, f'mae_vit_{size}_patch16')
    # infer_folder(mae_model_ft, dut_omron, save_folder=f'visualization/DUTS/ft_bddk_last_0.75', ratio=0.75)

    # bdd = '/Users/panda/Technion/datasets/bdd100k/images/100k/train'
    # ckpt_path_orig = '/Users/panda/Technion/TTA/ckpts/mae_visualize_vit_base.pth'
    # mae_model_orig = prepare_model(ckpt_path_orig, f'mae_vit_{size}_patch16')
    # infer_folder(mae_model_orig, bdd, save_folder=f'visualization/bddk_train/orig_0.75', ratio=0.75, n=15)
    # ckpt_path_ft = '/Users/panda/Technion/TTA/ckpts/best2.pth'
    # mae_model_ft = prepare_model(ckpt_path_ft, f'mae_vit_{size}_patch16')
    # infer_folder(mae_model_ft, bdd, save_folder=f'visualization/bddk_train/ft_best_0.75', ratio=0.75, n=15)
    # ckpt_path_ft2 = '/Users/panda/Technion/TTA/ckpts/last.pth'
    # mae_model_ft2 = prepare_model(ckpt_path_ft2, f'mae_vit_{size}_patch16')
    # infer_folder(mae_model_ft2, bdd, save_folder=f'visualization/bddk_train/ft_last_0.75', ratio=0.75, n=15)

    bdd = '/Users/panda/Technion/datasets/bdd100k/images/100k/val'
    ckpt_path_orig = '//ckpts/mae_visualize_vit_large_ganloss.pth'
    mae_model_orig = prepare_model(ckpt_path_orig, f'mae_vit_large_patch16')
    infer_folder(mae_model_orig, bdd, save_folder=f'visualization/bddk_train/orig_gan_0.75', ratio=0.75, n=30)

