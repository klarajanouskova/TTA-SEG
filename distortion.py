"""
Except fro adversarial perturbation, the methods are taken from
https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_imagenet_c.py

motion blur and snow removed because of wand dependency, may be added later

"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms.v2.functional import to_pil_image, to_tensor

from PIL import Image


# /////////////// Distortion Helpers ///////////////

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

from functools import partial

warnings.simplefilter("ignore", UserWarning)


def auc(errs):  # area under the alteration error curve
    area = 0
    for i in range(1, len(errs)):
        area += (errs[i] + errs[i - 1]) / 2
    area /= len(errs) - 1
    return area


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [60, 25, 12, 5, 3][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1):
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(224 - c[1], c[1], -1):
            for w in range(224 - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


# def barrel(x, severity=1):
#     c = [(0,0.03,0.03), (0.05,0.05,0.05), (0.1,0.1,0.1),
#          (0.2,0.2,0.2), (0.1,0.3,0.6)][severity - 1]
#
#     output = BytesIO()
#     x.save(output, format='PNG')
#
#     x = WandImage(blob=output.getvalue())
#     x.distort('barrel', c)
#
#     x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
#                      cv2.IMREAD_UNCHANGED)
#
#     if x.shape != (224, 224):
#         return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
#     else:  # greyscale to RGB
#         return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def fog(x, severity=1):
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    # x += c[0] * plasma_fractal(wibbledecay=c[1])[:x.shape[0], :x.shape[1]][..., np.newaxis]
    x += c[0] * cv2.resize(plasma_fractal(wibbledecay=c[1]), (x.shape[1], x.shape[0]))[..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1):
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost/frost1.png', './frost/frost2.png', './frost/frost3.png', './frost/frost4.jpeg', './frost/frost5.jpeg', './frost/frost6.jpeg'][idx]
    frost = cv2.imread(filename)
    # scale frost so that it's at least as big as the image, keep aspect ratio
    h, w, _ = frost.shape
    if h < w:
        if h <= x.size[0]:
            scale = (x.size[0] + 10) / h
            frost = cv2.resize(frost, (int(w * scale), int(h * scale)))
    else:
        if w <= x.size[1]:
            scale = (x.size[1] + 10) / w
            frost = cv2.resize(frost, (int(w * scale), int(h * scale)))

    y_start, x_start = np.random.randint(0, frost.shape[0] - x.size[1]), np.random.randint(0, frost.shape[1] - x.size[0])
    # cut and switch bgr to rgb
    frost = frost[y_start:y_start + x.size[1], x_start:x_start + x.size[0]][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def spatter(x, severity=1):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)
    x = x.convert('RGB')
    return x


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    # x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    # x = x.resize((224, 224), PILImage.BOX)
    x.resize((int(x.size[0] * c), int(x.size[1] * c)), PILImage.BOX)
    x.resize((x.size[0], x.size[1]), PILImage.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


def combine_distortions(severities=None, n=2):
    def pick_severity():
        # select a random severity
        if severities is None:
            severity = np.random.randint(0, 6)
        else:
            severity = np.random.choice(severities)
        return severity

    distortion_composed = None
    for i in range(n):
        new_distortion = np.random.choice(list(distortions.values()))
        severity = pick_severity()
        if severity == 0:
            # identity function
            distortion_composed = lambda x: x
        else:
            if distortion_composed is None:
                distortion_composed = partial(new_distortion, severity=severity)
            else:
                # compose distortion_composed function with new_distortion
                distortion_composed = partial(lambda x, f1, f2: f2(f1(x)), f1=distortion_composed, f2=new_distortion)

    return distortion_composed


"""
-------------------------------------------- 
ADVERSARIAL TRANSFORMATIONS

these assume an imagenet normalized torch tensor as input

--------------------------------------------
"""


def pgd(segmentation_model, ims_tensor, norm_fun, inv_norm_fun, iters=40, lr=0.005, thresh=0.4, epsilon=0.05,
        debug=False, gt='invert'):
    """
    Projected gradient descent adversarial attack, L_inf norm clipping
    Assumes segmentation model has frozen weights
    """

    assert gt in ['invert', 'random']

    ims_adv_tensor = ims_tensor.clone().detach()
    ims_adv_tensor.requires_grad = True
    optim = torch.optim.Adam([ims_adv_tensor], lr=lr)

    for it in range(iters):
        # Forward pass the data through the model
        seg_pred = segmentation_model.forward_seg(ims_adv_tensor, inference=True)

        if it == 0:
            if gt == 'random':
                # generate random gt
                r = (torch.rand_like(seg_pred) > thresh).float()
                # blur gt
                r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
                gt = seg_pred.clone()
                # gt[r > 0] = r[r > 0].detach()
                gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

            elif gt == 'invert':
                gt = (seg_pred < thresh).float()
            #     otherwise backward fails since we pass it multiple times
            gt = gt.detach()

            seg_preds_orig = seg_pred.clone()

        # Calculate the loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

        # Zero all existing gradients
        optim.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # perform optimization step
        optim.step()

        # clipping to max epsilon difference
        delta = ims_adv_tensor.detach() - ims_tensor
        delta_norm = torch.abs(delta)
        div = torch.clamp(delta_norm / epsilon, min=1.0)
        delta = delta / div
        ims_adv_tensor.data = ims_tensor + delta

        # visualize input image, adversarial image, their difference, and the segmentation masks
        if debug and (it % 5 == 0 or it == iters - 1):
            perturbed_images = ims_adv_tensor.clone()
            perturbed_images.detach()
            seg_pred_pert = segmentation_model.forward_seg(perturbed_images, inference=True)
            for img, img_adv, seg_orig, seg_adv in zip(ims_tensor, perturbed_images, seg_preds_orig, seg_pred_pert):
                # first show gt
                if it == 0:
                    plt.imshow(gt.squeeze().detach().cpu().numpy())
                    plt.title('gt')
                    plt.show()
                # run segmentation on the perturbed image
                plt.subplots(2, 3, figsize=(10, 6))
                plt.subplot(2, 3, 1)
                plt.imshow(to_pil_image(inv_norm_fun(img.cpu())))
                plt.title('input')
                plt.subplot(2, 3, 2)
                plt.imshow(to_pil_image(inv_norm_fun(img_adv.cpu())))
                plt.title('adversarial')
                plt.subplot(2, 3, 3)
                plt.imshow(to_pil_image(inv_norm_fun((img - img_adv).cpu())))
                plt.title('diff')
                plt.subplot(2, 3, 4)
                plt.imshow(seg_orig.squeeze().detach().cpu().numpy())
                plt.title('seg orig')
                plt.subplot(2, 3, 5)
                plt.imshow(seg_adv.squeeze().detach().cpu().numpy())
                plt.title('seg adv')
                pixels_changed = torch.sum((seg_orig > thresh) != (seg_adv > thresh))
                plt.suptitle(
                    f'Adversarial perturbation, eps: {epsilon}; lr: {lr} # iter: {it}, pixels changed: {pixels_changed}')
                # hide all axes
                for ax in plt.gcf().axes:
                    ax.axis('off')
                # increase title fonts
                for ax in plt.gcf().axes:
                    ax.title.set_fontsize(15)
                plt.show()

    # denormalize, clip the output, renormalize
    ims_adv_tensor = ims_adv_tensor.detach()
    perturbed_images = inv_norm_fun(ims_adv_tensor)
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    perturbed_images = norm_fun(perturbed_images)

    # Return the perturbed image
    return perturbed_images


def fgsm(segmentation_model, ims_tensor, norm_fun, inv_norm_fun, thresh=0.4, epsilon=0.05, debug=False, gt='invert'):
    """
    FGSM adversarial attack
    """

    assert gt in ['invert', 'random']

    # set model requires grad to false
    segmentation_model.freeze_seg_decoder()
    segmentation_model.freeze_encoder()

    ims_adv_tensor = ims_tensor.clone().detach()
    ims_adv_tensor.requires_grad = True

    # Forward pass the data through the model
    seg_pred = segmentation_model.forward_seg(ims_adv_tensor, inference=True)

    # generate random gt
    if gt == 'random':
        # generate random gt
        r = (torch.rand_like(seg_pred) > thresh).float()
        # blur gt
        r = torch.nn.functional.avg_pool2d(r, 5, stride=1, padding=2)
        gt = seg_pred.clone()
        # gt[r > 0] = r[r > 0].detach()
        gt = torch.clamp(r + (r * gt) * 0.5, 0, 1)

    elif gt == 'invert':
        gt = (seg_pred < thresh).float()

    # Calculate the loss
    # TODO test this without sigmpid applies
    loss = torch.nn.functional.binary_cross_entropy_with_logits(seg_pred, gt)

    # zero image gradients
    ims_adv_tensor.grad = None

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect ``datagrad``
    data_grad = ims_adv_tensor.grad.data
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_images = ims_adv_tensor + epsilon * sign_data_grad

    # no clipping here  since we have standardized images??

    # visualize input image, adversarial image, their difference, and the segmentation masks
    if debug:
        seg_pred_pert = segmentation_model.forward_seg(perturbed_images, inference=True)
        for img, img_adv, seg, seg_adv in zip(ims_tensor, perturbed_images, seg_pred, seg_pred_pert):
            # run segmentation on the perturbed image
            plt.subplots(2, 3, figsize=(10, 6))
            plt.subplot(2, 3, 1)
            plt.imshow(to_pil_image(inv_norm_fun(img.cpu())))
            plt.title('input')
            plt.subplot(2, 3, 2)
            plt.imshow(to_pil_image(inv_norm_fun(img_adv.cpu())))
            plt.title('adversarial')
            plt.subplot(2, 3, 3)
            plt.imshow(to_pil_image(inv_norm_fun((img - img_adv).cpu())))
            plt.title('diff')
            plt.subplot(2, 3, 4)
            plt.imshow(seg.squeeze().detach().cpu().numpy())
            plt.title('seg orig')
            plt.subplot(2, 3, 5)
            plt.imshow(seg_adv.squeeze().detach().cpu().numpy())
            plt.title('seg adv')
            pixels_changed = torch.sum((seg > thresh) != (seg_adv > thresh))
            plt.suptitle(
                f'Adversarial perturbation, eps: {epsilon}; pixels changed: {pixels_changed}')
            # hide all axes
            for ax in plt.gcf().axes:
                ax.axis('off')
            # increase title fonts
            for ax in plt.gcf().axes:
                ax.title.set_fontsize(15)
            plt.show()

    # denormalize, clip the output, renormalize
    perturbed_images = inv_norm_fun(ims_adv_tensor.detach())
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    perturbed_images = norm_fun(perturbed_images)

    # Return the perturbed image
    return perturbed_images.detach()

# make a dictionary with the name of the distortion as key and the function as value
# comment out those that change seg mask too much

distortions = OrderedDict([
    ('gaussian_noise', gaussian_noise),
    ('shot_noise', shot_noise),
    ('impulse_noise', impulse_noise),
    ('defocus_blur', defocus_blur),
    ('glass_blur', glass_blur),
    # ('zoom_Blur', zoom_blur),
    ('frost', frost),
    ('fog', fog),
    ('brightness', brightness),
    ('contrast', contrast),
    # ('elastic', elastic_transform),
    ('pixelate', pixelate),
    # ('jpeg', jpeg_compression),
    ('speckle_noise', speckle_noise),
    ('gaussian_blur', gaussian_blur),
    ('spatter', spatter)
    # doesn't look right
    # ('saturate', saturate)
])

distortions = OrderedDict([
    # ('gaussian_noise', gaussian_noise),
    ('shot_noise', shot_noise),
    # ('impulse_noise', impulse_noise),
    ('defocus_blur', defocus_blur),
    # ('glass_blur', glass_blur),
    # ('zoom_Blur', zoom_blur),
    ('frost', frost),
    ('fog', fog),
    ('brightness', brightness),
    ('contrast', contrast),
    # ('elastic', elastic_transform),
    # ('pixelate', pixelate),
    # ('jpeg', jpeg_compression),
    # ('speckle_noise', speckle_noise),
    # ('gaussian_blur', gaussian_blur),
    ('spatter', spatter)
    # doesn't look right
    # ('saturate', saturate)
])

# not used for validation
def distortions_test():
    distortions = OrderedDict([
        ('pixelate', pixelate),
        ('jpeg', jpeg_compression),
        ('saturate', saturate),
        ('combine', combine_distortions),
    ])


def visualize_distortion():
    import torchvision.transforms.v2.functional as F_tfm
    import torch.nn.functional as F
    import torchvision.io as io

    import matplotlib
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    inp_size = [1, 3, 384, 384]
    # inp = torch.rand(inp_size)

    # load the otter imahe and resize to inp_size
    inp = io.read_image('animals/cica.png').unsqueeze(0)
    if inp.shape[1] == 4:
        # remove alpha
        inp = inp[:, :3, :, :]
    inp = F.interpolate(inp, inp_size[2:])

    fig = plt.figure(constrained_layout=True, figsize=(5 * 3.7, len(distortions.keys()) * 3))
    subfigs = fig.subfigures(nrows=len(distortions.keys()), ncols=1)

    # show a grid - distortion severity in columns, distortion kind in rows
    for row, (subfig, (dist_name, dist_fun)) in enumerate(zip(subfigs, distortions.items())):
        for severity in range(1, 6):
            # subfig.suptitle(dist_name.replace('_', ' '), fontsize=40)
            ax = subfig.add_subplot(1, 5, severity)
            if severity == 1:
                ax.set_ylabel(dist_name.replace('_', ' '), fontsize=35)
            o = dist_fun(F_tfm.to_pil_image(inp[0]), severity)
            ax.imshow(np.array(o).astype(np.uint8))
            ax.tick_params(axis='both', which='both', length=0, labelsize=18)
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            # if row == len(distortions.keys()) - 1:
            # ax.set_xlabel(f'level {severity}', fontsize=35)
    plt.tight_layout()
    plt.show()


def test_distortion():
    import torchvision.transforms.v2.functional as F_tfm
    import torch.nn.functional as F
    import torchvision.io as io

    import matplotlib

    inp_size = [1, 3, 384, 384]
    # inp = torch.rand(inp_size)

    # load the otter imahe and resize to inp_size
    inp = io.read_image('animals/cica.png').unsqueeze(0)
    if inp.shape[1] == 4:
        # remove alpha
        inp = inp[:, :3, :, :]
    inp = F.interpolate(inp, inp_size[2:])

    o3 = jpeg_compression(F_tfm.to_pil_image(inp[0]), 3)
    o5 = jpeg_compression(F_tfm.to_pil_image(inp[0]), 5)
    plt.imshow(F_tfm.to_pil_image(inp[0]))
    plt.title('original')
    plt.show()
    plt.imshow(np.array(o3).astype(np.uint8))
    plt.title('distorted level 3')
    plt.show()
    plt.imshow(np.array(o5).astype(np.uint8))
    plt.title('distorted level 5')
    plt.show()
    print()


def get_random_corruption_fun(severities=None):
    # select a random distortion
    distortion = np.random.choice(list(distortions.values()))
    # select a random severity
    if severities is None:
        severity = np.random.randint(0, 6)
    else:
        severity = np.random.choice(severities)
    if severity == 0:
        # no distortion
        return lambda x: x
    # apply distortion
    return partial(distortion, severity=severity)



if __name__ == '__main__':
    # for severity in range(1, 6):
    # test_distortion()
    visualize_distortion()

