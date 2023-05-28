import timm
from torch import nn
import segmentation_models_pytorch as smp

from arg_composition import get_segmentation_args
from models_conv_mae import mae_vit_base_seg_conv_unet


class MaskLossNet(nn.Module):
    """
    Segmentation Loss Network that estimates the loss from the segmentation output, with a custom cnn backbone.
    """
    def __init__(self, size=(384, 384)):
        super(MaskLossNet, self).__init__()
        self.input_size = size
        #  conv layers followed by fully connected layers
        self.conv1 = nn.Conv2d(1, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(2, 2)
        self.output_size = (self.input_size[0] // 2 ** 4, self.input_size[1] // 2 ** 4)
        self.fc1 = nn.Linear(256 * self.output_size[0] * self.output_size[1], 64)
        self.fc2 = nn.Linear(64, 1)
        # 0-1 range loss such as IoU is assumed here
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = x.view(-1, 256 * self.output_size[0] * self.output_size[1])
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class MaskLossNetEff(MaskLossNet):
    """
    Segmentation Loss Network that estimates the loss from the segmentation output, with timm efficientnet backbone
    """
    def __init__(self):
        from timm.models.efficientnet import efficientnet_b0
        self.backbone = efficientnet_b0(pretrained=True)
        #      replace classification head with regression head
        self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        x = self.sigmoid(x)
        return x


class MaskLossDiscNet(MaskLossNet):
    """
    Segmentation Discriminator Network that estimates whether the mask comes from training domain or not.
    Same as MaskLossNet for now but targets should be binary.

    It should be trained together with the segmentation model eventually.
    """


class MaskLossUnet(nn.Module):
    """
    Network that predicts a corrected mask from the segmentation output so that segmentation loss is minimized.
    The architecture is a denoising autoencoder

    """
    def __init__(self, size=(384, 384)):
        super(MaskLossUnet, self).__init__()
        self.autoenc = smp.Unet(
            encoder_name="efficientnet-b0",  # choose a small encoder
            encoder_weights="imagenet",  # use `imagenet` pre-trained weighgit ts for encoder initialization
            in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            encoder_depth=5 # default and max is 5
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.autoenc(x)
        out = self.act(x)
        return out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


def test_loss_prediction():
    im, mask = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_seg = mae_vit_base_seg_conv_unet()
    model_seg.eval()
    model_loss = MaskLossNet()
    mse = nn.MSELoss()

    summary(model_loss, (1, 384, 384))
    with torch.no_grad():
        mask_pred = model_seg.forward_seg(im, inference=True)
    loss_pred = model_loss(mask_pred)
    iou = iou_loss(mask_pred, mask, reduction='none')
    loss = mse(iou, loss_pred)
    print()


def test_pred_discriminator():
    im, mask = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_seg = mae_vit_base_seg_conv_unet()
    model_seg.eval()
    model_disc = MaskLossDiscNet()
    mse = nn.MSELoss()

    summary(model_disc, (1, 384, 384))
    with torch.no_grad():
        mask_pred = model_seg.forward_seg(im, inference=True)
    loss_pred = model_disc(mask_pred)
    iou = iou_loss(mask_pred, mask, reduction='none')
    loss = mse(iou, loss_pred)
    print()


def test_mask_correction():
    im, mask = torch.randn(2, 3, 384, 384), (torch.randn(2, 1, 384, 384) > 0.5).float()

    model_seg = mae_vit_base_seg_conv_unet()
    model_seg.eval()
    model_corr = MaskLossUnet()
    mse = nn.MSELoss()

    summary(model_corr, (1, 384, 384))
    with torch.no_grad():
        mask_pred = model_seg.forward_seg(im, inference=True)
    mask_corr = model_corr(mask_pred)
    loss = mse(mask_corr, mask)
    print()


if __name__ == '__main__':
    import torch
    from torchsummary import summary
    from eval import iou_loss

    # test_loss_prediction()
    # test_pred_discriminator()
    test_mask_correction()

