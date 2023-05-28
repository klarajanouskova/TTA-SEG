# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------å------------------------------

from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed
from models_mae import MaskedAutoencoderSegViT, MaskedAutoencoderViT, EncoderViT, SegmentationDecoderViT, ReonstrucionDecoderViT


class MAExConv(MaskedAutoencoderSegViT):
    # TODO rewrite
    """
    The segmentation decoder will be a small conv net, instead of a transformer, should solve patch inconsistency issues.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0.):
        super(MaskedAutoencoderViT, self).__init__()
        # --------------------------------------------------------------------------
        # TTA encoder specifics

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # orig
        # self.blocks = nn.ModuleList([
        #     Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
        #     for i in range(depth)])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  drop_path=dpr[i])
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # TTA decoder specifics

        # common parts
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        # reconstruction part - keep names consistent with the reconstruction only model
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        # segmentation part
        self.tokens_to_pixels = nn.Linear(embed_dim, patch_size ** 2 * 32, bias=True)  # decoder to patch
        # TODO reshape
        self.decoder_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=5, stride=1, padding='same', bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same', bias=True),
            nn.ReLU(),
            # basically a linear layer
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.sigmoid = nn.Sigmoid()

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def forward_seg_decoder(self, x):
        # remove cls token
        # (N, P * P + 1, D) -> (N, P * P, D)
        x = x[:, 1:, :]

        # tokens to pixels
        # (N, L, P * P, D) -> (N, L, P * P, PS * PS)
        x = self.tokens_to_pixels(x)
        x = self.unpatchify(x, d=32)

        # includes final linear layer
        x = self.decoder_conv(x)

        return x

    def forward_seg(self, imgs, inference=False):
        latent, _, _ = self.forward_encoder(imgs, rec=False)
        pred = self.forward_seg_decoder(latent)  # [N, L, p*p*3]
        if inference:
            # # normalize to [0, 1], else this is part of loss (better for optim. in some cases)
            pred = pred.sigmoid()
        return pred


class MAExConvUnet(MaskedAutoencoderSegViT):
    """
    The segmentation decoder will be a small conv net, instead of a transformer,
     should solve patch inconsistency issues.
    """

    def __init__(self, img_size=384, patch_size=16, in_chans=3,
                 embed_dim=1024, encoder_depth=24, encoder_num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, drop_path_rate=0., unet_depth=3):
        super().__init__()

        self.encoder = EncoderViT(img_size, patch_size, in_chans, embed_dim, encoder_depth, encoder_num_heads,
                                  mlp_ratio,
                                  norm_layer, drop_path_rate)

        self.decoder_rec = ReonstrucionDecoderViT(self.encoder.num_patches, patch_size, in_chans, embed_dim,
                                                  decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio,
                                                  norm_layer)

        self.unet_channels = [32, 64, 128, 128][:unet_depth + 1]

        self.tokens_to_pixels = nn.Linear(embed_dim, patch_size ** 2 * self.unet_channels[0],
                                          bias=True)
        self.decoder_seg = Unet(self.unet_channels)

        self.sigmoid = nn.Sigmoid()

        self.norm_pix_loss = norm_pix_loss

        self.mse_loss = nn.MSELoss(reduction='none')

    def forward_seg(self, imgs, inference=False):
        x, _, _ = self.encoder.forward(imgs, rec=False)
        # remove cls token
        # (N, P * P + 1, D) -> (N, P * P, D)
        x = x[:, 1:, :]

        # tokens to pixels
        # (N, P * P, D) -> (N, L, P * P, PS * PS)
        x = self.tokens_to_pixels(x)
        x = self.unpatchify(x, d=self.unet_channels[0])

        pred = self.decoder_seg.forward(x)  # [N, L, p*p*3]
        if inference:
            # # normalize to [0, 1], else this is part of loss (better for optim. in some cases)
            pred = pred.sigmoid()
        return pred

    def train_norm_layers_only(self):
        """
        Freeze all layers except for the normalization layers - layernorm in transformer, batchnorm in convnets.
        Usefull for example for the TENT TTA method.
        """

        for name, p in self.named_parameters():
            if 'norm' in name.lower():
                p.requires_grad = True
            else:
                p.requires_grad = False




class Conv2dBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Unet(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.unet_channels = channels

        self.down_blocks = self.build_down_blocks()

        self.unet = UnetDecoder(
            encoder_channels=self.unet_channels,
            decoder_channels=self.unet_channels[2:],
            final_channels=self.unet_channels[-1],
            center=False
        )

        self.seg_pred = nn.Conv2d(self.unet_channels[-1], 1, kernel_size=1)

        self.sigmoid = nn.Sigmoid()

    def build_down_blocks(self):
        down_blocks = []
        n_blocks = len(self.unet_channels) - 1
        for i in range(n_blocks):
            scale_factor = 1 / 4**(n_blocks-i-1)
            down_blocks.append(
                UnetEncoderBlock(self.unet_channels[i], self.unet_channels[i + 1], scale_factor=scale_factor)
            )
        return nn.ModuleList(down_blocks)

    def forward(self, x):
        # includes final linear layer
        skips = []
        for block in self.down_blocks:
            x, skip = block(x)
            skips.append(skip)
        x = self.unet(skips)
        x = self.seg_pred(x)
        return x


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)

    def forward(self, x, skip: Optional[torch.Tensor] = None):
        if self.scale_factor != 1.0:
            # TODO this may be wrong?
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2.0, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        conv_args = dict(kernel_size=3, padding=1, act_layer=act_layer)
        self.scale_factor = scale_factor
        if norm_layer is None:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, **conv_args)
        else:
            self.conv1 = Conv2dBnAct(in_channels, out_channels, norm_layer=norm_layer, **conv_args)
            self.conv2 = Conv2dBnAct(out_channels, out_channels, norm_layer=norm_layer, **conv_args)
        self.down = Conv2dBnAct(out_channels, out_channels, kernel_size=5, stride=self.scale2stride(scale_factor),
                                padding=2, act_layer=act_layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_skip = self.down(x)
        return x, x_skip

    def scale2stride(self, scale_factor):
        return int(1 / scale_factor)


class UnetDecoder(nn.Module):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            norm_layer=nn.BatchNorm2d,
            center=False,
    ):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = UnetDecoderBlock(channels, channels, scale_factor=1.0, norm_layer=norm_layer)
        else:
            self.center = nn.Identity()

        in_channels = []

        # first just first encoder block (idx1, idx0 is in_channels), then decoder ones
        up_stream = [encoder_channels[1]] + list(decoder_channels)
        skip_stream = list(encoder_channels[2:])
        for in_chs, skip_chs in zip(up_stream, skip_stream):
            in_channels.append(in_chs + skip_chs)
        out_channels = decoder_channels

        self.blocks = nn.ModuleList()
        for in_chs, out_chs in zip(in_channels, out_channels):
            self.blocks.append(UnetDecoderBlock(in_chs, out_chs, scale_factor=4, norm_layer=norm_layer))
        self.final_conv = nn.Conv2d(out_chs, final_channels, kernel_size=(1, 1))

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x: List[torch.Tensor]):
        encoder_head = x[0]
        skips = x[1:]
        x = self.center(encoder_head)
        for i, b in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = b(x, skip)
        x = self.final_conv(x)
        return x


def mae_vit_base_patch16_seg_conv(**kwargs):
    model = MAExConv(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_seg_conv_unet(**kwargs):
    model = MAExConvUnet( embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    im, mask = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model = mae_vit_base_seg_conv_unet()
    model.train_norm_layers_only()
    # model = mae_vit_base_patch16()
    # model(im) # right now not working (parent forward not compatible)

    # loss_rec, pred_rec, mask_rec = model.forward_rec(im, mask_ratio=0.75)
    pred_seg = model.forward_seg(im)
    print(im.shape, pred_seg.shape)
