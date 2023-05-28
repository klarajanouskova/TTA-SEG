import matplotlib.pyplot as plt

from tta import *
from util.voc_dataset_seg import CA_CLEAN_IDXS_VAL_A

from eval import iou_loss


eval_distortions = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur',
                       'gaussian_blur', 'brightness', 'contrast', 'none']


def test_refinement(args, thresh=0.4, samples=10):
    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    tta_methods = ['rec', 'ref']
    # tta_methods = ['rec', 'iou']
    args.tta_grad_clip = -0.1
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[1] * len(tta_methods))


    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./tta_analysis'):
        os.mkdir('./tta_analysis')
    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
    for corr in ['spatter', 'brightness', 'contrast', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    # for corr in ['frost', 'fog', 'shot_noise', 'glass_blur']:
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        # create corr folder if it doesn't exist
        if not os.path.exists(f'./tta_analysis/{corr}'):
            os.mkdir(f'./tta_analysis/{corr}')

        for sev in [5]:

            # TODO handle idxs properly, sort them by TTA?
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[:samples]):
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int().to(device)

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting
                    dist = denorm_im.cpu().numpy().transpose(1, 2, 0)

                im_tensor_clean = img[None].float().to(device)

                pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)
                loss_ref_dist, mask_ref_dist = tta.get_refinement_loss(pred_seg_dist, return_mask=True)
                loss_ref_clean, mask_ref_clean = tta.get_refinement_loss(pred_seg_clean, return_mask=True)

                seg_loss_clean = iou_loss(pred_seg_clean, gt.repeat(len(pred_seg_clean), 1, 1, 1), thresh, apply_sigmoid=False)
                seg_loss_dist = iou_loss(pred_seg_dist, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh, apply_sigmoid=False)
                mask_loss_ref_clean = iou_loss(mask_ref_clean, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh, apply_sigmoid=False)
                mask_loss_ref_dist = iou_loss(mask_ref_dist, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh, apply_sigmoid=False)


                #  visualized im, distorted_im, seg. predictions and refinement masks
                plt.subplots(3, 2, figsize=(10, 15))
                plt.subplot(3, 2, 1)
                plt.imshow(denorm_im.cpu().numpy().transpose(1, 2, 0))
                plt.title('clean')
                plt.subplot(3, 2, 2)
                plt.imshow(dist)
                plt.title('distorted')
                plt.subplot(3, 2, 3)
                plt.imshow(pred_seg_clean.squeeze().cpu().numpy())
                plt.title('segmentation clean')
                plt.subplot(3, 2, 4)
                plt.imshow(pred_seg_dist.squeeze().cpu().numpy())
                plt.title('segmentation dist')
                plt.subplot(3, 2, 5)
                plt.imshow(mask_ref_clean.squeeze().cpu().numpy())
                plt.title('refined mask clean')
                plt.subplot(3, 2, 6)
                plt.imshow(mask_ref_dist.squeeze().cpu().numpy())
                plt.title('refined mask dist')

                # set all axis off
                for ax in plt.gcf().axes:
                    ax.set_axis_off()

                #     save the plot
                plt.savefig(f'./tta_analysis/{corr}/{sev}_{c}.png', bbox_inches='tight', pad_inches=0)
                # plt.show()

                print()

            #     so the plots don't consume too much memory when running on the server
            plt.close('all')


def tta_x_iou(args, thresh=0.4, samples=10):
    def plot_results(name, img, gt, preds_seg, preds_rec, dist_loss_estimates,  seg_iou_losses, loss_dict,
                     thresh=0.4, save_name=None, folder=''):
        tta_method = '&'.join(loss_dict.keys())
        # get the other losses aggregation from loss dict, convert to numpy, scale when possible
        seg_iou_losses = np.array(seg_iou_losses) * 100
        dist_loss_estimates = np.array(dist_loss_estimates) * 100
        # TODO fix this
        # tta_losses = np.array([l for l in loss_dict.values()]).sum(0)

        vis_im = im_to_vis(img)
        it_axis = np.arange(len(seg_iou_losses))

        r, c = len(preds_seg) + 1, 3
        plt.subplots(nrows=r, ncols=c, figsize=(c * 4, r * 4))

        baseline_pred = (preds_seg[0].squeeze() > thresh).int()

        for i, (pred_seg, pred_rec) in enumerate(zip(preds_seg, preds_rec)):
            pred_seg = pred_seg.squeeze()
            pred_seg_t = (pred_seg > thresh).int()
            pixel_idxs_better = np.where((pred_seg_t == gt) & (baseline_pred != gt))
            pixel_idxs_worse = np.where((pred_seg_t != gt) & (baseline_pred == gt))
            pred_seg_error_vis = pred_seg_t[:, :, None].numpy().repeat(3, axis=2)
            # make improved pixels green
            pred_seg_error_vis[pixel_idxs_better] = [0, 1, 0]
            # make worsened pixels red
            pred_seg_error_vis[pixel_idxs_worse] = [1, 0, 0]

            pred_seg_error_vis = pred_seg_error_vis * 255

            # tta_impr = (tta_losses[i] - tta_losses[0]) / tta_losses[0] if i > 0 else 1
            seg_impr = (seg_iou_losses[i] - seg_iou_losses[0]) / seg_iou_losses[0] if i > 0 else 1

            # seg
            plt.subplot(r, c, i * c + 1)
            plt.imshow(pred_seg_error_vis)
            color = 'red' if seg_impr > 0 else 'green'
            if seg_impr == 0:
                color = 'black'
            plt.title(f'{i} TTA: {seg_iou_losses[i]:.3f}, it0: {seg_impr:.3f}', color=color)
            plt.axis('off')

            # raw seg pred
            plt.subplot(r, c, i * c + 2)
            plt.imshow(pred_seg)
            plt.title('Segmentation prediction')
            plt.axis('off')

            # rec
            plt.subplot(r, c, i * c + 3)
            plt.imshow(im_to_vis(pred_rec))
            # color = 'red' if tta_impr > 0 else 'green'
            # if tta_impr == 0:
            #     color = 'black'
            # plt.title(f'{i} TTA: {tta_losses[i]:.3f}, it0: {tta_impr:.3f}', color=color)
            # plt.axis('off')

        # raw seg pred
        plt.subplot(r, c, (i + 1) * c + 1)
        plt.imshow(vis_im)
        plt.title('Input image')
        plt.axis('off')

        plt.subplot(r, c, (i + 1) * c + 2)
        plt.imshow(gt, cmap='gray')
        plt.title(f'Ground truth')
        plt.axis('off')

        # plt.subplot(r, c, (i + 1) * c + 3)
        # plt.plot(it_axis, (seg_iou_losses - seg_iou_losses[0]) / seg_iou_losses[0], label='Segmentation MRE')
        # # plt.plot(it_axis, (tta_losses - tta_losses[0]) / tta_losses[0], label=f'TTA - {tta_method}')
        # plt.plot(it_axis, (dist_loss_estimates - dist_loss_estimates[0]) / dist_loss_estimates[0],
        #          label=f'Distortion estimate')
        # # for loss_name, losses_vals in loss_dict.items():
        # #     losses_vals = np.array(losses_vals)
        # #     plt.plot(it_axis, (losses_vals - losses_vals[0]) / losses_vals[0], label=f'TTA - {loss_name}')
        # plt.xticks(it_axis)
        # plt.title('Mean Relative metrics over SSL iterations')
        # plt.xlabel('SSL iteration')
        # plt.ylabel('MRE')
        # plt.legend()

        plt.subplot(r, c, (i + 1) * c + 3)
        plt.plot(it_axis, seg_iou_losses, label='Segmentation MRE')
        # plt.plot(it_axis, (tta_losses - tta_losses[0]) / tta_losses[0], label=f'TTA - {tta_method}')
        plt.plot(it_axis, dist_loss_estimates,
                 label=f'Distortion estimate')
        # for loss_name, losses_vals in loss_dict.items():
        #     losses_vals = np.array(losses_vals)
        #     plt.plot(it_axis, (losses_vals - losses_vals[0]) / losses_vals[0], label=f'TTA - {loss_name}')
        plt.xticks(it_axis)
        plt.title('Metrics over SSL iterations')
        plt.xlabel('SSL iteration')
        plt.ylabel('Error')
        plt.legend()

        # make dir if it doesn't exist
        os.makedirs(f'pascal_tta/{folder}', exist_ok=True)
        if save_name is not None:
            plt.savefig(f'pascal_tta/{folder}{name}_{save_name}.jpg')

        plt.show()

        print()


    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    # we need to load iou for the esitmation and for example rec for ta
    tta_methods = ['rec', 'ref']
    # tta_methods = ['rec', 'iou']
    args.tta_grad_clip = -0.1
    args.tta_iter_num = 10
    rec_lr = 5e-2
    ref_lr = 1e-4
    rec_weight = 1
    # adjust weight so that we can use rec lr
    ref_weight = ref_lr / rec_lr
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[rec_weight, ref_weight], load_iou=True)

    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./tta_analysis'):
        os.mkdir('./tta_analysis')
    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
    # for corr in ['glass_blur', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    for corr in ['frost', 'fog', 'shot_noise', 'glass_blur']:
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        # create corr folder if it doesn't exist
        if not os.path.exists(f'./tta_analysis/{corr}'):
            os.mkdir(f'./tta_analysis/{corr}')

        for sev in [5]:

            # TODO handle idxs properly, sort them by TTA?
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[:samples]):
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int()

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor(
                        (dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting
                    dist = denorm_im.cpu().numpy().transpose(1, 2, 0)

                im_tensor_clean = img[None].float().to(device)

                # pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                # pred_seg_clean = pred_seg_clean.detach().cpu()
                # pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)

                # nim x it x c x h x w
                xs_tta, preds_seg_tta, preds_rec_tta, loss_dict = tta(im_tensor_dist)

                # remove nim dimension, as nim=1, or loop over them, it will be the batch dimension
                preds_seg_tta = preds_seg_tta.squeeze(0)
                preds_rec_tta = preds_rec_tta.squeeze(0)
                distortion_loss_estimate = tta.get_iou_loss(preds_seg_tta.to(device)).squeeze().detach().cpu().numpy()
                # distortion_loss_true = iou_loss(preds_seg_tta, (pred_seg_clean > thresh).repeat(len(preds_seg_tta), 1, 1, 1),
                #                                 thresh, apply_sigmoid=False).squeeze().detach().cpu().numpy()

                seg_loss = iou_loss(preds_seg_tta, gt.repeat(len(preds_seg_tta), 1, 1, 1), thresh, apply_sigmoid=False)\
                    .squeeze().detach().cpu().numpy()
                print()

                save_name = f'{sev}_{c}'
                folder = f'{corr}/'
                plot_results(name, img, gt.squeeze(), preds_seg_tta, preds_rec_tta, distortion_loss_estimate,
                             seg_loss, loss_dict, save_name=save_name, folder=folder)

            #     so the plots don't consume too much memory when running on the server
            plt.close('all')


def visualize_tta(args, thresh=0.4):
    def plot_results(name, img, gt, preds_seg, preds_rec,  seg_iou_losses, loss_dict,
                     thresh=0.4, save_name=None, folder=''):
        # increase title fontsize
        # get the other losses aggregation from loss dict, convert to numpy, scale when possible
        seg_iou_losses = np.array(seg_iou_losses) * 100
        # TODO fix this
        # tta_losses = np.array([l for l in loss_dict.values()]).sum(0)

        # vis_im = im_to_vis(img)
        vis_im = img
        it_axis = np.arange(len(seg_iou_losses))

        r, c = int(np.ceil(len(preds_seg) / 2)) + 1, 1
        fig, axs = plt.subplots(nrows=r, ncols=c, figsize=(c * 3.3, r * 3))

        for ax in axs.flatten():
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0, labelsize=18)

        baseline_pred = (preds_seg[0].squeeze() > thresh).int()

        # input image
        plt.subplot(r, c, 1)
        plt.imshow(vis_im)
        # plt.title('Input')
        plt.ylabel('Input', fontsize=22)
        # ground truth
        # plt.subplot(r, c, 2)
        # plt.imshow(gt, cmap='gray')
        # plt.title('GT')
        # plt.axis('off')


        for i, (pred_seg, pred_rec) in enumerate(zip(preds_seg, preds_rec)):
            if i % 2 != 0:
                continue
            pred_seg = pred_seg.squeeze()
            pred_seg_t = (pred_seg > thresh).int()
            pixel_idxs_better = np.where((pred_seg_t == gt) & (baseline_pred != gt))
            pixel_idxs_worse = np.where((pred_seg_t != gt) & (baseline_pred == gt))
            pred_seg_error_vis = pred_seg_t[:, :, None].numpy().repeat(3, axis=2)
            # make improved pixels green
            pred_seg_error_vis[pixel_idxs_better] = [0, 1, 0]
            # make worsened pixels red
            pred_seg_error_vis[pixel_idxs_worse] = [1, 0, 0]

            pred_seg_error_vis = pred_seg_error_vis * 255

            # tta_impr = (tta_losses[i] - tta_losses[0]) / tta_losses[0] if i > 0 else 1
            seg_impr = (seg_iou_losses[i] - seg_iou_losses[0]) if i > 0 else 0

            # seg
            plt.subplot(r, c, i // 2 + 2)
            plt.imshow(pred_seg_error_vis)
            color = 'red' if seg_impr > 0 else 'green'
            if seg_impr == 0:
                color = 'black'
            plt.title(f'{seg_iou_losses[i]:.2f} ({seg_impr:.2f})', color=color, fontsize=22)
            plt.ylabel(f'Iteration {i}', fontsize=22)

            # make dir if it doesn't exist
        # os.makedirs(f'pascal_tta/{folder}', exist_ok=True)
        # if save_name is not None:
        #     plt.savefig(f'thesis_vis/{folder}{name}_{save_name}.jpg')

        plt.tight_layout()
        # plt.subplots_adjust(hspace=0.1, wspace=0.5)
        plt.show()

        print()

    np.random.seed(0)
    torch.random.manual_seed(0)

    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    # we need to load iou for the esitmation and for example rec for ta
    tta_methods = ['rec', 'ref']
    # more doesn't fit
    args.tta_iter_num = 8
    rec_lr = 5e-2
    args.tta_lr = rec_lr
    rec_weight = 1
    # adjust weight so that we can use rec lr
    ref_weight = 0.001
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[rec_weight, ref_weight], load_iou=True)

    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./thesis_vis'):
        os.mkdir('./thesis_vis')
    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
    # for corr in ['glass_blur', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    # for corr in ['frost', 'fog', 'shot_noise', 'glass_blur']:
    for corr in ['frost', 'contrast', 'spatter']:
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        # create corr folder if it doesn't exist
        if not os.path.exists(f'./thesis_vis/{corr}'):
            os.mkdir(f'./thesis_vis/{corr}')

        for sev in [3]:

            # TODO handle idxs properly, sort them by TTA?
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[14:19]):
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int()

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor(
                        (dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting
                    dist = denorm_im.cpu().numpy().transpose(1, 2, 0)

                im_tensor_clean = img[None].float().to(device)

                # pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                # pred_seg_clean = pred_seg_clean.detach().cpu()
                # pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)

                # nim x it x c x h x w
                xs_tta, preds_seg_tta, preds_rec_tta, loss_dict = tta(im_tensor_dist)

                # remove nim dimension, as nim=1, or loop over them, it will be the batch dimension
                preds_seg_tta = preds_seg_tta.squeeze(0)
                preds_rec_tta = preds_rec_tta.squeeze(0)
                # distortion_loss_true = iou_loss(preds_seg_tta, (pred_seg_clean > thresh).repeat(len(preds_seg_tta), 1, 1, 1),
                #                                 thresh, apply_sigmoid=False).squeeze().detach().cpu().numpy()

                seg_loss = iou_loss(preds_seg_tta, gt.repeat(len(preds_seg_tta), 1, 1, 1), thresh, apply_sigmoid=False)\
                    .squeeze().detach().cpu().numpy()
                print()

                save_name = f'{sev}_{c}'
                folder = f'{corr}/'
                plot_results(name, dist, gt.squeeze(), preds_seg_tta, preds_rec_tta,
                             seg_loss, loss_dict, save_name=save_name, folder=folder)

            #     so the plots don't consume too much memory when running on the server
            # plt.close('all')


def test_iou(args, thresh=0.4, samples=10):
    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    tta_methods = ['rec', 'ref']
    # tta_methods = ['rec', 'iou']
    args.tta_grad_clip = -0.1
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[1] * len(tta_methods))


    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./tta_analysis'):
        os.mkdir('./tta_analysis')
    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
        # for corr in ['glass_blur', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    for corr in ['frost', 'fog', 'shot_noise', 'glass_blur']:
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        # create corr folder if it doesn't exist
        if not os.path.exists(f'./tta_analysis/{corr}'):
            os.mkdir(f'./tta_analysis/{corr}')

        for sev in [5]:

            # TODO handle idxs properly, sort them by TTA?
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[:samples]):
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int()

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting
                    dist = denorm_im.cpu().numpy().transpose(1, 2, 0)

                im_tensor_clean = img[None].float().to(device)

                pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)

                # iou_loss(im_preds_seg, gt.repeat(len(im_preds_seg), 1, 1, 1), thresh, apply_sigmoid=False)

                print()

            #     so the plots don't consume too much memory when running on the server
            plt.close('all')


def eval_refinement_iou(args, thresh=0.4, samples=20):
    np.random.seed(0)
    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    tta_methods = ['rec', 'ref']
    # tta_methods = ['rec', 'iou']
    args.tta_grad_clip = -0.1
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[1] * len(tta_methods))

    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./tta_analysis'):
        os.mkdir('./tta_analysis')
    # for corr in ['spatter', 'shot_noise', 'fog', 'brightness', 'contrast', 'frost']:
    # for corr in ['glass_blur', 'defocus_blur', 'gaussian_noise', 'gaussian_blur', 'none']:
    dist_losses_base, dist_losses_ref = [], []
    clean_losses_base, clean_losses_ref = [], []
    # for corr in ['frost', 'fog', 'shot_noise', 'glass_blur']:
    for corr in ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur', 'gaussian_blur', 'brightness', 'contrast', 'none']:
        dist_corr_losses_base, dist_corr_losses_ref = [], []
        clean_corr_losses_base, clean_corr_losses_ref = [], []
        if corr == 'none':
            corrupt_fun = None
        else:
            corrupt_fun = distortions[corr]

        if corrupt_fun is None:
            corr = 'clean'

        # create corr folder if it doesn't exist
        if not os.path.exists(f'./tta_analysis/{corr}'):
            os.mkdir(f'./tta_analysis/{corr}')

        for sev in [1, 3, 5]:
            print(f'{corr} - {sev}')
            dist_sev_losses_base, dist_sev_losses_ref = [], []
            clean_sev_losses_base, clean_sev_losses_ref = [], []
            # TODO handle idxs properly, sort them by TTA?
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[:samples]):
                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                gt = (gt > 0).int().to(device)

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting

                im_tensor_clean = img[None].float().to(device)

                pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)
                loss_ref_dist, mask_ref_dist = tta.get_refinement_loss(pred_seg_dist, return_mask=True)
                loss_ref_clean, mask_ref_clean = tta.get_refinement_loss(pred_seg_clean, return_mask=True)

                seg_loss_clean = iou_loss(pred_seg_clean, gt.repeat(len(pred_seg_clean), 1, 1, 1), thresh,
                                          apply_sigmoid=False)
                seg_loss_dist = iou_loss(pred_seg_dist, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh,
                                         apply_sigmoid=False)
                mask_loss_ref_clean = iou_loss(mask_ref_clean, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh,
                                               apply_sigmoid=False)
                mask_loss_ref_dist = iou_loss(mask_ref_dist, gt.repeat(len(pred_seg_dist), 1, 1, 1), thresh,
                                              apply_sigmoid=False)

                dist_sev_losses_base.append(seg_loss_dist.item())
                dist_sev_losses_ref.append(mask_loss_ref_dist.item())
                clean_sev_losses_base.append(seg_loss_clean.item())
                clean_sev_losses_ref.append(mask_loss_ref_clean.item())

                # print(f'Segmentation loss clean: {seg_loss_clean.item()}; refined: {mask_loss_ref_clean.item()}')
                # print(f'clean {"improved" if mask_loss_ref_clean.item() < seg_loss_clean.item() else "worsened"}')
                # print(f'Segmentation loss dist: {seg_loss_dist.item()}; refined: {mask_loss_ref_dist.item()}')
                # print(f'distorted {"improved" if mask_loss_ref_dist.item() < seg_loss_dist.item() else "worsened"}')

            dist_corr_losses_ref.append(dist_sev_losses_ref)
            dist_corr_losses_base.append(dist_sev_losses_base)
            clean_corr_losses_ref.append(clean_sev_losses_ref)
            clean_corr_losses_base.append(clean_sev_losses_base)
        dist_losses_base.append(dist_corr_losses_base)
        dist_losses_ref.append(dist_corr_losses_ref)
        clean_losses_base.append(clean_corr_losses_base)
        clean_losses_ref.append(clean_corr_losses_ref)
    dist_losses_base = np.array(dist_losses_base)
    dist_losses_ref = np.array(dist_losses_ref)
    clean_losses_base = np.array(clean_losses_base)
    clean_losses_ref = np.array(clean_losses_ref)

    print('Mean IoU - dist base')
    print(100 - 100 * dist_losses_base.mean())
    print('Mean IoU - dist ref')
    print(100 - 100 * dist_losses_ref.mean())
    print('Mean IoU - clean base')
    print(100 - 100 * clean_losses_base.mean())
    print('Mean IoU - clean ref')
    print(100 - 100 * clean_losses_ref.mean())

    # save results to file
    folder = 'refinement_corr'
    # create folder
    if not os.path.exists(f'./tta_analysis/{folder}'):
        os.mkdir(f'./tta_analysis/{folder}')
    np.save(f'./tta_analysis/{folder}/dist_losses_base.npy', dist_losses_base)
    np.save(f'./tta_analysis/{folder}/dist_losses_ref.npy', dist_losses_ref)
    np.save(f'./tta_analysis/{folder}/clean_losses_base.npy', clean_losses_base)
    np.save(f'./tta_analysis/{folder}/clean_losses_ref.npy', clean_losses_ref)


def eval_iou_accuracy(args, thresh=0.4, samples=20):
    def plot_ious(real, estimated, corr, sev, save=False):
        # first sort real and estimated by real, descending
        real, estimated = zip(*sorted(zip(real, estimated), reverse=True))
        real, estimated = np.array(real), np.array(estimated)
        #   normalize iou by mean and std
        real_normalized = (real - real.mean()) / real.std()
        estimated_normalized = (estimated - estimated.mean()) / estimated.std()

        #   plot normalized real and estimaed ious in one graph, their difference and non-normalized in otehrs
        plt.subplots(1, 3, figsize=(20, 10))
        plt.subplot(1, 3, 1)
        plt.title(f'Normalized real and estimated IoU for {corr}/{sev} images')
        plt.plot(real_normalized, label='real', color='darkgreen')
        plt.plot(estimated_normalized, label='estimated', color='dodgerblue')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.title(f'Real and estimated IoU for {corr}/{sev} images')
        plt.plot(real, label='real', color='darkgreen')
        plt.plot(estimated, label='estimated', color='dodgerblue')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.title(f'Difference between real and estimated IoU for {corr}/{sev} images')
        plt.plot(real - estimated, label='real - estimated', color='crimson')
        plt.legend()


        if save:
            plt.savefig(f'./tta_analysis/{corr}/iou_est_{corr}_{sev}.png')

        plt.show()
        plt.close()

        print()

    #     fix seed for reproducibility
    np.random.seed(0)

    # first plot normalized values, real and predicted IoU side by side
    args.data_cls_sub = 'dog&cow&bird&cow'
    dataset = get_pascal(args, split='val')

    tta_methods = ['rec', 'iou']
    # tta_methods = ['rec', 'iou']
    args.tta_grad_clip = -0.1
    tta = TestTimeAdaptor(args=args, tta_method='&'.join(tta_methods), weights=[1] * len(tta_methods))

    severities = [5, 3, 1]

    # create tta_analysis folder if it doesn't exist
    if not os.path.exists('./tta_analysis'):
        os.mkdir('./tta_analysis')

    estimated_ious_all, real_ious_all = [], []

    for sev in severities:
        estimated_ious_sev, real_ious_sev = [], []

        # for corr in eval_distortions[:-1]: # exclude none distortion
        for corr in eval_distortions:
            if corr == 'none':
                # only do clean images once
                if corr != 5:
                    continue
                corrupt_fun = None
            else:
                corrupt_fun = distortions[corr]

            if corrupt_fun is None:
                corr = 'clean'

            # create corr folder if it doesn't exist
            if not os.path.exists(f'./tta_analysis/{corr}'):
                os.mkdir(f'./tta_analysis/{corr}')

            # TODO handle idxs properly, sort them by TTA?
            estimated_ious, real_ious = [], []
            for c, idx in enumerate(CA_CLEAN_IDXS_VAL_A[:samples]):

                # TODO return idx of box within image so that we can identify them uniquely?
                img, gt, cls, name = dataset[idx]

                denorm_im = img * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                            torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)

                # distort and add batch dimension
                if corrupt_fun is not None:
                    dist = corrupt_fun(to_pil_image(denorm_im), severity=sev) / 255
                    #   renormalize, go back to tensor
                    dist_img = to_tensor((dist - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD))

                    im_tensor_dist = dist_img[None].float().to(device)
                else:
                    im_tensor_dist = img[None].float().to(device)
                    # for plotting

                im_tensor_clean = img[None].float().to(device)

                # TODO check whether we should use GT or predicted clean, same with threshold, depending on args
                pred_seg_clean = tta.forward_segmentation(im_tensor_clean, inference=True)
                pred_seg_dist = tta.forward_segmentation(im_tensor_dist, inference=True)

                loss_true = iou_loss(pred_seg_dist, pred_seg_clean > thresh, thresh,
                                          apply_sigmoid=False)
                loss_estimate = tta.get_iou_loss(pred_seg_dist)
                estimated_ious.append(loss_estimate.item())
                real_ious.append(loss_true.item())

            estimated_ious_sev.extend(estimated_ious)
            real_ious_sev.extend(real_ious)

            real_ious = np.array(real_ious) * 100
            estimated_ious = np.array(estimated_ious) * 100

            # for given severity and distortion
            plot_ious(real_ious, estimated_ious, corr=corr, sev=sev, save=True)

        estimated_ious_all.extend(estimated_ious_sev)
        real_ious_all.extend(real_ious_sev)

        estimated_ious_sev = np.array(estimated_ious_sev) * 100
        real_ious_sev = np.array(real_ious_sev) * 100

        corr = 'all'
        if not os.path.exists(f'./tta_analysis/{corr}'):
            os.mkdir(f'./tta_analysis/{corr}')
        # for given severity across all distortions
        plot_ious(real_ious_sev, estimated_ious_sev, corr=corr, sev=sev, save=True)

    # for all severities across all distortions
    real_ious_all = np.array(real_ious_all) * 100
    estimated_ious_all = np.array(estimated_ious_all) * 100
    sev_str = ','.join(str(s) for s in severities)
    # create corr folder if it doesn't exist
    corr = 'all'
    if not os.path.exists(f'./tta_analysis/{corr}'):
        os.mkdir(f'./tta_analysis/{corr}')
    plot_ious(real_ious_all, estimated_ious_all, corr=corr, sev=sev_str, save=True)






if __name__ == '__main__':
    args = get_segmentation_args(inference=True).parse_args()
    # most recent model to test - adversarial trained
    args.tta_iou_model_run = 'deep_loss_adv_clean_gt_soft_method_qual_min_sev0_segloss_IoU_trainloss_0.0005_l1'
    args.tta_ref_model_run = 'deep_loss_adv_clean_gt_soft_method_ref_min_sev0_segloss_IoU_trainloss_0.0005'
    # corruption trained
    # args.tta_iou_model_run = 'deep_loss_clean_gt_method_qual_min_sev0_segloss_IoU_trainloss_0.0001_l1'
    # args.tta_ref_model_run = 'deep_loss_clean_gt_method_ref_min_sev0_segloss_IoU_trainloss_0.0005'
    print()
    # eval_iou_accuracy(args)
    # good for visualizing tta process
    # tta_x_iou(args)
    # test_refinement(args)
    # eval_refinement_iou(args)
    visualize_tta(args)