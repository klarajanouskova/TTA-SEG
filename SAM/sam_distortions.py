import torch
import matplotlib.pyplot as plt



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

        # zero image gradients
        ims_adv_tensor.grad = None

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