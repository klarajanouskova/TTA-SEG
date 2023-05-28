"""
Inference time self-supervised learning
"""

import sys
import os

import torch
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
# prevent matpltolib form using scientific notation
plt.rcParams['axes.formatter.useoffset'] = False

# use latex font
plt.rc('text', usetex=True)


from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from arg_composition import get_segmentation_args

sys.path.append('..')

local = not torch.cuda.is_available()
device = 'cpu' if local else 'cuda'


def plot_results_all(rec_mses, seg_deep_losses, seg_iou_losses, save_name=None, folder=''):
    r, c = 2, 2
    plt.subplots(nrows=r, ncols=c, figsize=(c * 4, r * 4))

    plt.subplot(r, c, 1)
    it_axis = np.arange(len(seg_deep_losses))
    plt.plot(it_axis, seg_deep_losses, label='Segmentation learnt loss', color='darkgreen')
    plt.xticks(it_axis)
    plt.title('Segmentation loss evolution over SSL iterations')
    plt.xlabel('SSL iteration')
    plt.ylabel('Deep Loss')
    plt.legend()

    plt.subplot(r, c, 2)
    it_axis = np.arange(len(seg_iou_losses))
    plt.plot(it_axis, seg_iou_losses, label='Segmentation IoU', color='dodgerblue')
    plt.xticks(it_axis)
    plt.title('Segmentation error evolution over SSL iterations')
    plt.xlabel('SSL iteration')
    plt.ylabel('IoU Loss')
    plt.legend()

    plt.subplot(r, c, 3)
    plt.plot(it_axis, rec_mses, label='Reconstruction MSE', color='indigo')
    plt.xticks(it_axis)
    plt.title('Reconstruction error over SSL iterations')
    plt.xlabel('SSL iteration')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(r, c, 4)
    plt.plot(it_axis, (seg_iou_losses - seg_iou_losses[0]) / seg_iou_losses[0], label='Segmentation MRE', color='dodgerblue')
    plt.plot(it_axis, (rec_mses - rec_mses[0]) / rec_mses[0], label='Reconstruction MRE', color='indigo')
    plt.plot(it_axis, (seg_deep_losses - seg_deep_losses[0]) / seg_deep_losses[0],
             label='Segmentation deep loss MRE', color='darkgreen')
    plt.xticks(it_axis)
    plt.title('Mean Relative Error over SSL iterations')
    plt.xlabel('SSL iteration')
    plt.ylabel('MRE')
    plt.legend()

    # make dir if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    if save_name is not None:
        plt.suptitle(save_name)
        plt.savefig(os.path.join(folder, f'{save_name}.jpg'))

    plt.show()

def im_to_vis(im):
    denorm_im = im * torch.tensor(IMAGENET_DEFAULT_STD).view(3, 1, 1) + \
                         torch.tensor(IMAGENET_DEFAULT_MEAN).view(3, 1, 1)
    im = F.to_pil_image(denorm_im)
    return im


def denormalize(img):
    """
    Transform image-net normalized image to original [0, 1] range
    """
    return img * torch.tensor(IMAGENET_DEFAULT_STD)[:, None, None] + torch.tensor(IMAGENET_DEFAULT_MEAN)[:, None, None]


def results_to_readable(folder='pascal_tta_small_distorted', pre='distortion', kinds=[]):
    # np.save(f'{folder}/{save_name}.npy', [rec_mses, seg_deep_losses, seg_iou_losses])

    severities = [5, 3, 1]
    lrs = [5e-2, 1e-2, 5e-3]
    # evaluate reconstruction finetuning only
    for severity in severities:
        best_results = []

        for lr in lrs:
            save_name = f'lr_{lr}_bs_8_its_21_rw_0.0_sw_1.0_sev_{severity}'

            # load the results for given severity and learning rate
            if not os.path.exists(f'{folder}/{save_name}.npy'):
                print(f'File {save_name} does not exist')
                best_results.append(None)
            #     3 x kind x sample x it
            rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)

            print(f'Results for: {save_name}\n\n')
            print('rec_mse & seg_deep_loss & seg_iou_loss\n')

            # get best iteration based on iou:
            best_iter = np.mean(seg_iou_losses, axis=(0, 1)).argmin()
            print(f'{np.mean(rec_mses[:, :, best_iter]):.3f} & {np.nanmean(seg_deep_losses[:, :, best_iter]):.3f} & {np.mean(seg_iou_losses[:, :, best_iter]):.3f}\n')
            print(f'Best iteration: {best_iter}\n')
            # now print the results for each distortion
            for dist, dist_rec_mses, dist_seg_deep_losses, dist_seg_iou_losses in zip(distortion_keys, rec_mses,
                                                                                      seg_deep_losses, seg_iou_losses):
                print(f'{dist}: {np.mean(dist_rec_mses[:, best_iter]):.3f} & {np.nanmean(dist_seg_deep_losses[:, best_iter]):.3f} & {np.mean(dist_seg_iou_losses[:, best_iter]):.3f}\n')

            # also log which percentage of images has improved iou for each iteration
            print('\n\n')
            impr = seg_iou_losses / seg_iou_losses[:, :, 0][:, :, None] < 1
            # improvement in latex format
            impr_str = ' & '.join(
                [f'{np.sum(impr[:, :, i]) / impr.shape[0] / impr.shape[1]:.1f}' for i in range(impr.shape[2])])
            print(f'Improvement in iou for each iteration: {impr_str}\n')

            print('Improvement in iou for each iteration per distortion\n')

            for dist, dist_seg_iou_losses in zip(distortion_keys, seg_iou_losses):
                impr = dist_seg_iou_losses / dist_seg_iou_losses[:, 0][:, None] < 1
                impr_str = ' & '.join([f'{np.sum(impr[:, i]) / impr.shape[0]:.1f}' for i in range(impr.shape[1])])
                print(f'{dist}: {impr_str}\n')

            #     also print all iou results aggregated over samples and distortions
            print(f'Iou results aggregated over samples and distortions:\n')
            print(f'{np.mean(seg_iou_losses, axis=(0, 1))}\n')
            # now print the results for each category
            for cat, cat_seg_iou_losses in zip(distortion_keys, seg_iou_losses):
                print(f'{cat}: {np.mean(cat_seg_iou_losses, axis=0)}\n')

            # also create plots and save them for easier analysis of what is going on at different iterations
            # plot_results_all(rec_mses.mean(axis=(0, 1)), seg_deep_losses.mean(axis=(0, 1)), seg_iou_losses.mean(axis=(0, 1)),
            #                  save_name=save_name, folder=folder)

            # plot results for each distortion
            for dist, dist_rec_mses, dist_seg_deep_losses, dist_seg_iou_losses in zip(distortion_keys, rec_mses,
                                                                                      seg_deep_losses, seg_iou_losses):
                save_folder = f'{folder}/{dist}'
                plot_results_all(dist_rec_mses.mean(axis=0), dist_seg_deep_losses.mean(axis=0),
                                 dist_seg_iou_losses.mean(axis=0), save_name=save_name, folder=save_folder)


def per_distortion_lr_plot(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):
    # TODO one for each corruption
    rows = []
    columns_names = ['corruption', 'level', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$',
                     'diff abs', 'diff cor rel', 'diff cor total', 'best it', 'best lr']
    severities = [5, 3, 1]
    # rec lrs
    rec_lrs = [1e-1, 7e-2, 5e-2, 1e-2, 5e-3]
    # tent lrs
    tent_lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # iou lrs
    # ioucorr_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    iou_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # iou_lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # ref lrs
    # refcorr_lrs = [1e-1, 5e-2,  1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    ref_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # ref_lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # adv lrs
    adv_lrs = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

    # evaluate reconstruction finetuning only
    freeze_rec = True

    method_names = {'rec': 'REC', 'tent': 'ENT', 'iou': 'dIoU', 'ref': 'REF', 'adv': 'ADV'}
    methods = ['rec', 'tent', 'iou', 'ref', 'adv']
    # create subplot for each method
    fig, axs = plt.subplots(2, int(np.ceil(len(methods) / 2)), figsize=(len(methods) * 3, 10))
    flat_axs = axs.flatten()

    # get the 'twilight' colormap list
    cmap = plt.cm.get_cmap('viridis')
    # get the colors for each method
    colors = [cmap(i) for i in np.linspace(0, 0.9, max(len(lrs) for lrs in [rec_lrs, tent_lrs, iou_lrs, ref_lrs, adv_lrs]))]


    for m_id, (method, method_lrs) in enumerate(zip(methods, [rec_lrs, tent_lrs, iou_lrs, ref_lrs, adv_lrs])):
        results = []
        for severity in severities:
            lrs_results = []
            for lr in method_lrs:
                #                 # tta_rec_lr_0.01_freeze_rec_1_its_10_sev_5.npy
                save_name = f'tta_{method}_lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'
                # save_name = f'tta_adv_lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                lrs_results.append(seg_iou_losses)

            results.append(lrs_results)
        # sev x lr x kind x sample x it
        results = np.array(results)
        print()

        none_base = 100 - results[0, 0, -1, :, 0].mean()
        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(0, 2, 3))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = overall_res[np.arange(best_it_per_lr.shape[0]), best_it_per_lr].argmin()

        # plot the results in subplot
        for lr_idx, lr_res in enumerate(overall_res):
            flat_axs[m_id].plot(lr_res, color=colors[lr_idx], linewidth=2)

        # increase tick size
        flat_axs[m_id].tick_params(axis='both', which='major', labelsize=18)
        flat_axs[m_id].set_title(f'{method_names[method]}', size=22)
        flat_axs[m_id].set_xlabel('iteration', fontsize=23)
        flat_axs[m_id].set_ylabel('IoU (\%)', fontsize=23)
        flat_axs[m_id].set_ylim([21.5, 26])
        flat_axs[m_id].set_xlim([0, 10])
        #     legend
        flat_axs[m_id].legend([f'lr={lr:1.0e}' for lr in method_lrs], loc='upper left', fontsize=17, ncol=2)
    # make the last subplot blank
    flat_axs[-1].axis('off')

    # show color map in the last subplot
    # cbar_ax = flat_axs[-1]
    # # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, shrink=0.1)
    # # plot cmap in half ot the last subplot
    # cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), cax=cbar_ax, shrink=0.5)
    # cbar.ax.tick_params(labelsize=17)
    # cbar.set_label('learning rate', fontsize=20)


    # plt.suptitle(f'Loss over TTA iterations with different learning rates')
    plt.tight_layout()
    plt.show()



def per_distortion_table_best_res(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):

    rows = []
    columns_names = ['corruption', 'level', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$',
                     'diff abs', 'diff cor rel', 'diff cor total', 'best it', 'best lr']
    severities = [5, 3, 1]
    # rec lrs
    # lrs = [1e-1, 7e-2, 5e-2, 1e-2, 5e-3]
    # tent lrs
    # lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # iou lrs
    # lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # ioucorr lrs
    # lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    # ref lrs
    lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # refcorr lrs
    # lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    # adv lrs
    # lrs = [1e-1, 5e-1, 5e-2, 1e-2, 5e-3, 1e-3]


    # evaluate reconstruction finetuning only

    freeze_rec = True

    results = []
    for severity in severities:
        lrs_results = []
        for lr in lrs:
            #                 # tta_rec_lr_0.01_freeze_rec_1_its_10_sev_5.npy
            save_name = f'tta_ref_lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'
            # load the results for given severity and learning rate
            if not os.path.exists(f'{folder}/{save_name}.npy'):
                print(f'File {save_name} does not exist')
            #     3 x kind x sample x it
            # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
            deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
            lrs_results.append(seg_iou_losses)

        results.append(lrs_results)
    # sev x lr x kind x sample x it
    results = np.array(results)
    print()

    none_base = 100 - results[0, 0, -1, :, 0].mean()
    #  results aggregated over samples and distortions, lr x it
    overall_res = results.mean(axis=(0, 2, 3))
    best_it_per_lr = overall_res.argmin(1)
    best_lr_idx = overall_res[np.arange(best_it_per_lr.shape[0]), best_it_per_lr].argmin()
    base = 100 - overall_res[0, 0]
    best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
    rows += [['all', f'{", ".join(str(s) for s in severities)}', base, best,
              best - base, (best - base) / (none_base - base) * 100, (best - base) / overall_res[0, 0] * 100,
              best_it_per_lr[best_lr_idx], lrs[best_lr_idx]]]

    #      now add a row for each corruption and severity separately
    for kind_idx, kind in enumerate(kinds):
        for sev_idx, severity in enumerate(severities):
            if kind == 'none' and sev_idx > 0:
                continue
            #   lr  x it
            res = results[sev_idx, :, kind_idx, :, :].mean(1)
            best_it_per_lr = res.argmin(1)
            best_lr_idx = res[np.arange(best_it_per_lr.shape[0]), best_it_per_lr].argmin()
            base = 100 - res[0, 0]
            best = 100 - res[best_lr_idx, best_it_per_lr[best_lr_idx]]
            rows += [[kind.replace('_', ' '), severity, base, best,  best - base,
                      (best - base) / (none_base - base) * 100, (best - base) / res[0, 0] * 100,
                      best_it_per_lr[best_lr_idx], lrs[best_lr_idx]]]

    #   now print the results as latex table
    table = pd.DataFrame(rows, columns=columns_names)
    # cast iteration to integer
    # print(table.to_latex(index=False, header=True, float_format="%.2f"))

    # shuffle rows a bit and add some more formatting
    # s = table.to_latex(index=False, header=True, float_format="%.2f")

    # table['best lr'] = table['best lr'].astype(str)
    # alternative - g removes trailing zeros
    table['best lr'] = table['best lr'].map(lambda x: '%1.5g' % x)

    styled_table = table.style. \
        hide(axis="index"). \
        format(precision=2)
    # print(styled_table.to_latex(hrules=True))
    s = styled_table.to_latex(hrules=True)

    t = s.split("\\\\")
    # header, midrule, all, none, midrule
    header_and_general_str = " \\\\ ".join([t[0]] + [t[1]] + t[-2:-1]) + '\\\\ \n\\midrule \n'
    footer = " \\\\ ".join(t[-1:])
    corrs = t[2:-2]
    #     join corrs by new line in latex and every third row by \midrule
    corrs_str = "\n \midrule".join([' \\\\ '.join(corrs[i:i + len(severities)]) + ' \\\\ ' for i in range(0, len(corrs), 3)])
    composed_str = header_and_general_str + " \\\\ ".join([corrs_str + footer])
    print(composed_str)


def per_distortion_table_best_res_comb(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):

    rows = []
    columns_names = ['corruption', 'level', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$', 'diff abs', 'corr err diff (%)', 'total err diff(%)', 'best it', 'best w2']
    severities = [5, 3, 1]
    # rec + ref
    lr = 5e-2
    w1 = 1
    # w2s = [0.01, 0.002, 0.001, 0.0002]

    w2s = [0.02]
    w3s = [0.01]

    # evaluate reconstruction finetuning only

    freeze_rec = True

    results = []
    for severity in severities:
        w2_results = []
        for w2 in w2s:
            for w3 in w3s:
                # save_name = f'tta_rec&refcorr_lr_{lr}_ws_{w1}_{w2}_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'
                save_name = f'tta_rec&ref&iou_lr_{lr}_ws_{w1}_{w2}_{w3}_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_75_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                # w2_results.append(seg_iou_losses)
                results.append(seg_iou_losses)

        # results.append(w2_results)
    # sev x lr x kind x sample x it
    # results = np.array(results)
    results = np.array(results)[:, None]


    none_base = 100 - results[0, 0, -1, :, 0].mean()
    #  results aggregated over samples and distortions, lr x it
    overall_res = results.mean(axis=(0, 2, 3))
    best_it_perw2 = overall_res.argmin(1)
    best_w2_idx = overall_res[np.arange(best_it_perw2.shape[0]), best_it_perw2].argmin()
    base = 100 - overall_res[0, 0]
    best = 100 - overall_res[best_w2_idx, best_it_perw2[best_w2_idx]]
    rows += [['all', f'{", ".join(str(s) for s in severities)}', base, best,  best - base,
              (best - base) / (none_base - base) * 100, (best - base) / overall_res[0, 0] * 100,
              best_it_perw2[best_w2_idx], w2s[best_w2_idx]]]

    #      now add a row for each corruption and severity separately
    for kind_idx, kind in enumerate(kinds):
        for sev_idx, severity in enumerate(severities):
            if kind == 'none' and sev_idx > 0:
                continue
            #   lr  x it
            res = results[sev_idx, :, kind_idx, :, :].mean(1)
            best_it_perw2 = res.argmin(1)
            best_w2_idx = res[np.arange(best_it_perw2.shape[0]), best_it_perw2].argmin()
            base = 100 - res[0, 0]
            best = 100 - res[best_w2_idx, best_it_perw2[best_w2_idx]]
            rows += [[kind.replace('_', ' '), severity, base, best,  best - base,
                      (best - base) / (none_base - base) * 100, (best - base) / res[0, 0] * 100,
                      best_it_perw2[best_w2_idx], w2s[best_w2_idx]]]

    #   now print the results as latex table
    table = pd.DataFrame(rows, columns=columns_names)

    # alternative - g removes trailing zeros
    table['best w2'] = table['best w2'].map(lambda x: '%1.5g' % x)

    styled_table = table.style. \
        hide(axis="index"). \
        format(precision=2)
    # print(styled_table.to_latex(hrules=True))
    s = styled_table.to_latex(hrules=True)

    t = s.split("\\\\")
    # header, midrule, all, none, midrule
    header_and_general_str = " \\\\ ".join([t[0]] + [t[1]] + t[-2:-1]) + '\\\\ \n\\midrule \n'
    footer = " \\\\ ".join(t[-1:])
    corrs = t[2:-2]
    #     join corrs by new line in latex and every third row by \midrule
    corrs_str = "\n \midrule".join([' \\\\ '.join(corrs[i:i + len(severities)]) + ' \\\\ ' for i in range(0, len(corrs), 3)])
    composed_str = header_and_general_str + " \\\\ ".join([corrs_str + footer])
    print(composed_str)


def compare_results_freeze_rec(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):
    rows = []
    columns_names = ['frozen rec. dec.', 'IoU', 'TTA IoU', 'abs', 'base', 'total', 'it', 'lr']

    severities = [5, 3, 1]
    lrs = [1e-1, 5e-2, 1e-2, 5e-3]
    # evaluate reconstruction finetuning only

    freeze_rec_vals = [True, False]

    for freeze_rec in freeze_rec_vals:
        results = []
        # aggregate results
        for severity in severities:
            lrs_results = []
            for lr in lrs:
                save_name = f'tta_rec_lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                lrs_results.append(seg_iou_losses)

            results.append(lrs_results)

        # sev x lr x kind x sample x it
        results = np.array(results)

        none_base = 100 - results[0, 0, -1, :, 0].mean()

        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(0, 2, 3))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = overall_res[np.arange(len(lrs)), best_it_per_lr].argmin()
        base = 100 - overall_res[0, 0]
        best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
        rows += [[freeze_rec, base, best, best - base, (best - base) / (none_base - base) * 100,
                  (best - base) / overall_res[0, 0] * 100, best_it_per_lr[best_lr_idx], lrs[best_lr_idx]]]

    #   now print the results as latex table
    table = pd.DataFrame(rows, columns=columns_names)
    # cast iteration to integer
    print(table.to_latex(index=False, header=True, float_format="%.2f"))


def compare_results_ttas_grad_clip(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):

    severities = [5, 3, 1]
    # lrs = [5e-2, 1e-2, 5e-3]
    method_lrs = {'tent': [5e-3], 'rec': [5e-2], 'ref': [5e-4], 'iou': [5e-4]}
    method = 'tent'
    lrs = method_lrs[method]
    gc_vals = [-1, 0.5, 1., 1.5, 3.]
    columns_names = ['corruption', 'level', 'mIoU_base'] + [f'clip {gc}' if gc > 0 else 'no clip' for gc in gc_vals]

    # evaluate reconstruction finetuning only

    tmp_res = [[] for _ in range(3 + len(gc_vals))]
    for i, grad_clip in enumerate(gc_vals):
        results = []
        # aggregate results
        for severity in severities:
            lrs_results = []
            for lr in lrs:
                save_name = f'tta_{method}_lr_{lr}_freeze_rec_1_its_10_gradclip_{grad_clip}_nims_1_sev_{severity}'
                # save_name = f'lr_{lr}_freeze_rec_{int(freeze_rec)}_its_10_rw_0.0_sw_1.0_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                lrs_results.append(seg_iou_losses)

            results.append(lrs_results)

        # sev x lr x kind x sample x it
        results = np.array(results)

        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(0, 2, 3))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = 0
        if i == 0:
            tmp_res[0] += ['all']
            tmp_res[1] += ['{3, 4, 5}']
            base = 100 - overall_res[0, 0]
            tmp_res[2] += [base]
        best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
        tmp_res[i + 3] += [best]

        #     now for each
        for kind_idx, kind in enumerate(kinds):
            for sev_idx, severity in enumerate(severities):
                if kind == 'none' and sev_idx > 0:
                    continue
                #   lr  x it
                res = results[sev_idx, :, kind_idx, :, :].mean(1)
                best_it_per_lr = res.argmin(1)
                best_lr_idx = 0
                base = 100 - res[0, 0]
                best = 100 - res[best_lr_idx, best_it_per_lr[best_lr_idx]]
                if i == 0:
                    tmp_res[0] += [kind.replace('_', ' ')]
                    tmp_res[1] += [severity]
                    tmp_res[2] += [base]
                tmp_res[i + 3] += [best]


    #   now print the results as latex table
    table = pd.DataFrame(np.array(tmp_res).T, columns=columns_names)
    # cast numbers to floats
    for col in columns_names[2:]:
        table[col] = table[col].astype(float)
    styled_table = table.style.\
        highlight_max(subset=columns_names[3:], props='textbf:--rwrap;', axis=1).\
        hide(axis="index").\
        format(precision=2)
    print(styled_table.to_latex(hrules=True))


def compare_results_ttas_nims(folder='pascal_tta_clean_120_distorted', pre='distortion', kinds=[]):

    severities = [5, 3, 1]
    # lrs = [5e-2, 1e-2, 5e-3]
    lrs = [5e-2]
    nim_vals = [1, 3, 5, 8]
    columns_names = ['corruption', 'level', 'mIoU_base'] + nim_vals

    # evaluate reconstruction finetuning only

    tmp_res = [[] for _ in range(3 + len(nim_vals))]
    for i, nim_val in enumerate(nim_vals):
        results = []
        # aggregate results
        for severity in severities:
            lrs_results = []
            for lr in lrs:
                save_name = f'tta_tent_lr_{lr}_freeze_rec_1_its_10_gradclip_-1_nims_{nim_val}_sev_{severity}'
                # save_name = f'lr_{lr}_freeze_rec_{int(freeze_rec)}_its_10_rw_0.0_sw_1.0_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                lrs_results.append(seg_iou_losses)

            results.append(lrs_results)

        # sev x lr x kind x sample x it
        results = np.array(results)

        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(0, 2, 3))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = 0
        if i == 0:
            tmp_res[0] += ['all']
            tmp_res[1] += ['{3, 4, 5}']
            base = 100 - overall_res[0, 0]
            tmp_res[2] += [base]
        best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
        tmp_res[i + 3] += [best]

        #     now for each
        for kind_idx, kind in enumerate(kinds):
            for sev_idx, severity in enumerate(severities):
                if kind == 'none' and sev_idx > 0:
                    continue
                #   lr  x it
                res = results[sev_idx, :, kind_idx, :, :].mean(1)
                best_it_per_lr = res.argmin(1)
                best_lr_idx = 0
                base = 100 - res[0, 0]
                best = 100 - res[best_lr_idx, best_it_per_lr[best_lr_idx]]
                if i == 0:
                    tmp_res[0] += [kind.replace('_', ' ')]
                    tmp_res[1] += [severity]
                    tmp_res[2] += [base]
                tmp_res[i + 3] += [best]

    #   now print the results as latex table
    table = pd.DataFrame(np.array(tmp_res).T, columns=columns_names)
    # cast numbers to floats
    for col in columns_names[2:]:
        table[col] = table[col].astype(float)
    styled_table = table.style.\
        highlight_max(subset=columns_names[3:], props='textbf:--rwrap;', axis=1).\
        hide(axis="index").\
        format(precision=2)
    print(styled_table.to_latex(hrules=True))


def compare_results_ttas_methods(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):
    methods = ['rec', 'iou', 'ref', 'tent', 'adv']
    columns_names = ['corruption', 'level', 'base'] + [m for m in methods]

    severities = [5, 3, 1]
    # lrs = [5e-2, 1e-2, 5e-3]
    # high_lrs = [5e-2]
    # low_lrs = [0.0005]
    # method_lrs = {'tent': [5e-3], 'rec': [5e-2], 'refcorr': [5e-4], 'ioucorr': [5e-2], 'adv': [0.1]}
    method_lrs = {'tent': [5e-3], 'rec': [5e-2], 'ref': [1e-3], 'iou': [5e-4], 'adv': [0.1]}
    # evaluate reconstruction finetuning only

    tmp_res = [[] for _ in range(3 + len(methods))]
    for i, method in enumerate(methods):
        results = []
        # aggregate results
        for severity in severities:
            lrs_results = []
            lrs = method_lrs[method]
            for lr in lrs:
                save_name = f'tta_{method}_lr_{lr}_ws_1_freeze_rec_1_its_10_gradclip_-1_nims_1_sev_{severity}'
                # save_name = f'lr_{lr}_freeze_rec_{int(freeze_rec)}_its_10_rw_0.0_sw_1.0_sev_{severity}'

                # load the results for given severity and learning rate
                if not os.path.exists(f'{folder}/{save_name}.npy'):
                    print(f'File {save_name} does not exist')
                #     3 x kind x sample x it
                # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                lrs_results.append(seg_iou_losses)

            results.append(lrs_results)

        # sev x lr x kind x sample x it
        results = np.array(results)

        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(0, 2, 3))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = 0
        if i == 0:
            tmp_res[0] += ['all']
            tmp_res[1] += ['{3, 4, 5}']
            base = 100 - overall_res[0, 0]
            tmp_res[2] += [base]
        best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
        tmp_res[i + 3] += [best]

        #     now for each
        for kind_idx, kind in enumerate(kinds):
            for sev_idx, severity in enumerate(severities):
                if kind == 'none' and sev_idx > 0:
                    continue
                #   lr  x it
                res = results[sev_idx, :, kind_idx, :, :].mean(1)
                # we shoudl keep the overall best it here
                # best_it_per_lr = res.argmin(1)
                best_lr_idx = 0
                base = 100 - res[0, 0]
                best = 100 - res[best_lr_idx, best_it_per_lr[best_lr_idx]]
                if i == 0:
                    tmp_res[0] += [kind.replace('_', ' ')]
                    tmp_res[1] += [severity]
                    tmp_res[2] += [base]
                tmp_res[i + 3] += [best]


    #   now print the results as latex table
    table = pd.DataFrame(np.array(tmp_res).T, columns=columns_names)
    # cast numbers to floats
    for col in columns_names[2:]:
        table[col] = table[col].astype(float)

    styled_table = table.style. \
        highlight_max(subset=columns_names[3:], props='textbf:--rwrap;', axis=1). \
        hide(axis="index"). \
        format(precision=2)
    # print(styled_table.to_latex(hrules=True))
    s = styled_table.to_latex(hrules=True)

    t = s.split("\\\\")
    # header, midrule, all, none, midrule
    header_and_general_str = " \\\\ ".join([t[0]] + [t[1]] + t[-2:-1]) + '\\\\ \n\\midrule \n'
    footer = " \\\\ ".join(t[-1:])
    corrs = t[2:-2]
    #     join corrs by new line in latex and every third row by \midrule
    corrs_str = "\n \midrule".join([' \\\\ '.join(corrs[i:i + 3]) + ' \\\\ ' for i in range(0, len(corrs), 3)])
    composed_str = header_and_general_str + " \\\\ ".join([corrs_str + footer])
    print(composed_str)


def analyze_best_runs(folder='pascal_tta_clean_20_distorted', kinds=[]):
    methods = ['iou', 'ref', 'tent', 'rec', 'adv']
    method_lr = {'tent': 5e-3, 'rec': 5e-2, 'ref': 1e-3, 'iou': 5e-4, 'adv': 0.1}

    method_names = {'rec': 'REC', 'tent': 'ENT', 'iou': 'IOU', 'ref': 'REF', 'adv': 'ADV'}
    fig, axs = plt.subplots(len(methods), 3, figsize=(22, len(methods) * 4))
    flat_axs = axs.flatten()

    for m_idx, method in enumerate(methods):
        lr = method_lr[method]
        severity = 5
        save_name = f'tta_{method}_lr_{lr}_ws_1_freeze_rec_1_its_10_gradclip_-1_nims_1_sev_{severity}'
        deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)

        print()
        #     aggregate kinds x samples x its to (kinds x samples) x its
        seg_iou_losses = seg_iou_losses.reshape(-1, seg_iou_losses.shape[-1])
        # rel  > 0 means that the method improved the result
        relative_differences = (seg_iou_losses[:, 0] - seg_iou_losses[:, -1]) / seg_iou_losses[:, 0]
        abs_differences = seg_iou_losses[:, 0] - seg_iou_losses[:, -1]
        # sort seg losses by the first iteration
        sorted_idx = seg_iou_losses[:, 0].argsort()[::-1]
        sorted_losses = seg_iou_losses[sorted_idx]


        # line plot by sorted iou losses in first col
        axs[m_idx][0].plot(sorted_losses[:, 0], label='before TTA', color='indigo', linewidth=2)
        axs[m_idx][0].plot(sorted_losses[:, -1], label=f'after TTA', color='yellowgreen', linewidth=1, alpha=0.8)
        # axs[m_idx][0].set_xlabel('sample',  fontsize='20')
        # axs[m_idx][0].set_ylabel(f'{method_names[method]}: IoU loss (\%)', fontsize='20')
        axs[m_idx][0].set_ylabel(f'{method_names[method]}', fontsize=28)
        if m_idx == 0:
            axs[m_idx][0].set_title(f'Per-Image IoU Loss', size=28)
        axs[m_idx][0].tick_params(axis='both', which='major', labelsize=19)

        axs[m_idx][0].legend(loc='upper right', fontsize=19, ncol=2)

        #     histogram of relative differences in second col, make sure no bin contains both positive and negative values

        bins = 20
        # bins = list(np.arange(-20, 5, 5)) + list(np.arange(5, 25, 5))
        _, bin_edges, bars = axs[m_idx][1].hist(abs_differences, bins=bins)
        for bar in bars:
            if bar.get_x() < 0:
                if bar.get_x() + bar.get_width() < 0:
                    bar.set_facecolor("red")
                else:
                    bar.set_facecolor("blue")
            else:
                bar.set_facecolor("limegreen")
        axs[m_idx][1].tick_params(axis='both', which='major', labelsize=19)
        # set ticks for every second bin
        axs[m_idx][1].set_xticks(np.round(bin_edges[::4]))

        if m_idx == 0:
            axs[m_idx][1].set_title(f'IoU Improvement Histogram', size=28)

        #     plot number of improved / deteriorated samples in third col
        axs[m_idx][2].bar(['improved', 'deteriorated'], [np.sum(abs_differences > 0), np.sum(abs_differences < 0)],
                          color=['limegreen', 'red'])

        if m_idx == 0:
            axs[m_idx][2].set_title(f'IoU Change', size=28)
        axs[m_idx][2].tick_params(axis='both', which='major', labelsize=23)


    plt.legend()
    plt.tight_layout()
    plt.show()


def oracle_results(folder='pascal_tta_clean_20_distorted', kinds=[]):
    methods = ['iou', 'ref', 'tent', 'rec', 'adv']
    method_lr = {'tent': 5e-3, 'rec': 5e-2, 'ref': 1e-3, 'iou': 5e-4, 'adv': 0.1}

    method_names = {'rec': 'REC', 'tent': 'ENT', 'iou': 'dIoU', 'ref': 'REF', 'adv': 'ADV'}
    # make single figure, no subplots

    fig = plt.figure(figsize=(15, 10))

    bar_cols = []
    col_names = []

    cmap = plt.get_cmap('tab20b')
    # list of colors for each method
    methods_results = []
    for m_idx, method in enumerate(methods):
        act_results, act_losses = [], []
        lr = method_lr[method]

        for severity in [1, 3, 5]:
            save_name = f'tta_{method}_lr_{lr}_ws_1_freeze_rec_1_its_10_gradclip_-1_nims_1_sev_{severity}'
            deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)

            #     aggregate kinds x samples x its to (kinds x samples) x its
            seg_iou_losses = seg_iou_losses.reshape(-1, seg_iou_losses.shape[-1])
            act_losses.append(seg_iou_losses)
        # sort seg losses by the first iteration
        act_losses = np.vstack(act_losses)
        sorted_idx = act_losses[:, 0].argsort()[::-1]
        sorted_losses = act_losses[sorted_idx]
        act_results = np.array(sorted_losses)
        methods_results.append(act_results)

        # horizontal line with per-image mean iou loss before tta
        if m_idx == 0:
            plt.plot(act_results[:, 0], label='NA', color='indigo', linewidth=4)
            plt.axhline(y=act_results[:, 0].mean(), color='indigo', linestyle='--', linewidth=2, label='NA mean')
            bar_cols.append(act_results[:, 0].mean())
            col_names.append('NA')
        # horizontal line with per-image mean iou loss after tta
        # plt.axhline(y=seg_iou_losses[:, -1].mean(), color=colors[m_idx], linestyle='--', linewidth=2, label=f'{method_names[method]}')
        bar_cols.append(act_results[:, -1].mean())
        col_names.append(f'{method_names[method]}')

    # also load combination results
    comb_results = []
    for severity in [1, 3, 5]:
        save_name = f'tta_rec&refcorr_lr_0.05_ws_1_0.001_freeze_rec_1_its_10_gradclip_-1_nims_1_sev_{severity}'
        deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)

        #     aggregate kinds x samples x its to (kinds x samples) x its
        seg_iou_losses = seg_iou_losses.reshape(-1, seg_iou_losses.shape[-1])
        comb_results.append(seg_iou_losses)
    comb_results = np.vstack(comb_results)
    # sort by first iteration
    sorted_idx = comb_results[:, 0].argsort()[::-1]
    comb_results = comb_results[sorted_idx]

    # append comb results to bar cols
    bar_cols.append(comb_results[:, -1].mean())
    col_names.append('REC+REF')


    # method results: method x sample x iteration
    methods_results = np.array(methods_results)
    #  same line plot but with best method result for each sample
    best_results = methods_results[:, :, -1].min(axis=0)
    plt.plot(best_results, label=f'TTA oracle', color='crimson', linewidth=4, alpha=0.9)

    # mean hline
    plt.axhline(y=best_results.mean(), color='crimson', linestyle='--',
                label='oracle mean',linewidth=2)
    bar_cols.append(best_results.mean())
    col_names.append('oracle')

    #     best method and best iteration
    best_results = methods_results.min(axis=(0, 2))
    plt.plot(best_results, label=f'TTA oracle+', color='springgreen', linewidth=1, alpha=1)
    # mean hline
    plt.axhline(y=best_results.mean(), color='springgreen', linestyle='--',
                label='oracle+ mean', linewidth=2)
    bar_cols.append(best_results.mean())
    col_names.append('oracle+')

    plt.axhline(y=comb_results[:, -1].mean(), color='steelblue', linestyle='--',
                label=f'REC+REF mean',linewidth=2)

    plt.xlabel(f'Image', fontsize=32)
    plt.ylabel(f'IoU (\%)', fontsize=32)
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.legend(loc='upper right', fontsize=30, ncol=2)
    plt.tight_layout()
    plt.show()

    # sort bar cols and col names
    bar_cols = np.array(bar_cols)
    col_names = np.array(col_names)
    sorted_idx = bar_cols.argsort()[::-1]
    bar_cols = bar_cols[sorted_idx]
    col_names = col_names[sorted_idx]


    # bar plot
    fig = plt.figure(figsize=(15, 10))
    plt.bar(col_names, bar_cols, color=[cmap(i) for i in np.linspace(0, 1, len(bar_cols))])
    plt.ylim(bar_cols.min() - 2, bar_cols.max() + 1)
    plt.tick_params(axis='both', which='major', labelsize=36)
    plt.ylabel('IoU (\%)', fontsize=36)
    # add black edge to second bar, thicker
    plt.gca().patches[1].set_edgecolor('red')
    plt.gca().patches[1].set_linewidth(5)
    # make y vertical
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()








if __name__ == '__main__':
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur', 'gaussian_blur', 'brightness', 'contrast', 'none']
    # per_distortion_table_best_res(kinds=distortion_keys)
    # per_distortion_table_best_res_comb(kinds=distortion_keys)
    # get_agg_results_nim(kinds=distortion_keys)
    # compare_results_ttas_grad_clip(kinds=distortion_keys)
    # compare_results_ttas_methods(kinds=distortion_keys)
    # analyze_single_run(kinds=distortion_keys)
    # compare_results_freeze_rec(kinds=distortion_keys)
    # per_distortion_lr_plot(kinds=distortion_keys)
    oracle_results(kinds=distortion_keys)