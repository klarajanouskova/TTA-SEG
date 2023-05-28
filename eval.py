from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import models_mae, models_conv_mae

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'

#IoU, Structure and F Loss taken from ICON
# IoU Loss / to be paired with BCE!


def load_seg_model(args, pick='best'):
    assert pick in ['best', 'last']
    if local:
        model_path = f'ckpts/{args.model_name}-{pick}.pth'
    else:
        model_path = f'{args.output_dir}/{args.run_name}/checkpoint-{pick}.pth'

    ckpt = torch.load(model_path, map_location=device)

    if 'conv' in args.model:
        model = models_conv_mae.__dict__[args.model](img_size=args.input_size, unet_depth=args.unet_depth, patch_size=args.patch_size)
    else:
        model = models_mae.__dict__[args.model](img_size=args.input_size)

    model.to(device)

    msg = model.load_state_dict(ckpt['model'], strict=True)
    print(msg)
    return model


def load_tta_model(args, model_class, size=(384, 384), pick='best'):
    assert pick in ['best', 'last']
    if local:
        model_path = f'ckpts/{args.tta_model_run}-{pick}.pth'
    else:
        model_path = f'{args.output_dir}/{args.tta_model_run}/checkpoint-{pick}.pth'
    # deep_loss_sigmoid_segloss_IoU_trainloss_l1_0.0005

    ckpt = torch.load(model_path, map_location=device)

    model = model_class(size=size)
    model.to(device)
    msg = model.load_state_dict(ckpt['model'], strict=True)
    print(msg)
    return model


def iou_loss(pred, mask, threshold=None, reduction='none', apply_sigmoid=True):
    assert reduction in ['mean', 'none']
    # because we are training with BCEwithLogitsLoss so sigmoid is not applied to the output
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    if threshold:
        pred = (pred > threshold).float()
    inter = (pred*mask).sum(dim=(2, 3))
    union = (pred+mask).sum(dim=(2, 3))
    iou = 1-(inter+1)/(union-inter+1)
    if reduction == 'mean':
        return iou.mean()
    else:
        return iou


# Structure Loss
def structure_loss(pred, mask, reduction='mean', apply_sigmoid=True):
    assert apply_sigmoid is not False, 'F.binary_cross_entropy_with_logits assumes sigmoid was not applied to the output'

    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    if reduction == 'mean':
        return (wbce+wiou).mean()
    else:
        return wbce+wiou


# F Loss
def f_loss(pred, mask, beta=0.3, log_like=False, reduction='mean', apply_sigmoid=False):
    if apply_sigmoid:
        pred = torch.sigmoid(pred)
    eps = 1e-10
    n = pred.size(0)
    tp = (pred * mask).view(n, -1).sum(dim=1)
    h = beta * mask.view(n, -1).sum(dim=1) + pred.view(n, -1).sum(dim=1)
    fm = (1+beta) * tp / (h+eps)
    if log_like:
        floss = -torch.log(fm)
    else:
        floss = (1-fm)
    if reduction == 'mean':
        return floss.mean()
    else:
        return floss

# TODO rewrite reduction to have 'batch' option, which is what we want
def bce_wrapper(pred, mask, reduction='mean', apply_sigmoid=True):
    assert apply_sigmoid is not False, 'F.binary_cross_entropy_with_logits assumed sigmoid was not applied to the output'
    orig = F.binary_cross_entropy_with_logits(pred, mask, reduction=reduction)
    if reduction == 'none':
        # reduce over the spatial dimensions
        orig = orig.mean(dim=(2, 3))
    return orig


class Loss(nn.Module):
    valid_vals = ['STR', 'FL', 'IoU', 'BCE']
    losses_dict = {'BCE': bce_wrapper,
                   'IoU': iou_loss,
                   'IoU_25': partial(iou_loss, threshold=0.25),
                   'IoU_50': partial(iou_loss, threshold=0.5),
                   'IoU_75': partial(iou_loss, threshold=0.75),
                   'IoU_90': partial(iou_loss, threshold=0.9),
                   'STR': structure_loss,
                   'FL': f_loss,
                   }

    def __init__(self, loss_str, weights=None, reduction='mean', apply_sigmoid=True):
        self.losses = self.validate_and_parse_loss_str(loss_str)
        if weights is None:
            self.weights = [1] * len(self.losses)
        self.reduction = reduction
        self.apply_sigmoid = apply_sigmoid

    def __call__(self, pred, mask):
        loss_vals = [self.losses_dict[l](pred, mask, reduction=self.reduction, apply_sigmoid=self.apply_sigmoid) for l in self.losses]
        weighted_loss = sum([w * l for w, l in zip(self.weights, loss_vals)])
        loss_vals_dict = {loss: l for loss, l in zip(self.losses, loss_vals)}
        return weighted_loss, loss_vals_dict

    def validate_and_parse_loss_str(self, loss_str):
        parsed_loss = loss_str.split('+')
        for val in parsed_loss:
            if val not in self.valid_vals:
                raise ValueError('Invalid loss function: {}'.format(val))
        return parsed_loss