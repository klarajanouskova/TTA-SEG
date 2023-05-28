"""
Inference time self-supervised learning
"""

import sys
import os

import torch
import torchvision.transforms.v2.functional as F
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
# prevent matpltolib form using scientific notation
plt.rcParams['axes.formatter.useoffset'] = False


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


def per_distortion_lr_plot(folder='pascal_tta_clean_20_distorted', pre='distortion', kinds=[]):

    rows = []
    columns_names = ['corruption', 'level', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$',
                     'diff abs', 'diff cor rel', 'diff cor total', 'best it', 'best lr']
    severities = [5, 3, 1]
    # rec lrs
    rec_lrs = [1e-1, 7e-2, 5e-2, 1e-2, 5e-3]
    # tent lrs
    tent_lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # iou lrs
    # iou_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    iou_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # ref lrs
    # ref_lrs = [1e-1, 5e-2,  1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    ref_lrs = [1e-1, 5e-2,  1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    # adv lrs
    adv_lrs = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

    # evaluate reconstruction finetuning only

    freeze_rec = True

    methods = ['rec', 'tent', 'ioucorr', 'refcorr', 'adv']
    # create subplot for each method
    fig, axs = plt.subplots(1, len(methods), figsize=(len(methods) * 5, 5))

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
        for lr_res in overall_res:
            axs[m_id].plot(lr_res)

        axs[m_id].set_title(f'{method}')
        axs[m_id].set_xlabel('iteration')
        axs[m_id].set_ylabel('mIoU')
        # axs[m_id].set_ylim([0, 100])
        axs[m_id].set_xlim([0, 10])
        #     legend
        axs[m_id].legend([f'lr={lr}' for lr in method_lrs], loc='lower right')
    plt.suptitle(f'Loss over TTA iterations with different learning rates')
    plt.show()


def per_weather_table_best_res(folder='cityscapes_results/tta'):
    rows = []
    columns_names = ['weather condition', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$',
                     'diff abs', 'diff total']
    weather_conds = ['base', 'rain', 'fog']
    tent_lrs = [5e-3]
    sam_iou_lrs = [5e-4]
    ref_lrs = [1e-4]

    # evaluate reconstruction finetuning only

    lrs = ref_lrs
    for weather_cond in weather_conds:
        lrs_results = []
        for lr in lrs:
            # save_name = f'sam_tta_tent_lr_{lr}_ws_1.0_its_10_nims_1'
            save_name = f'ref_tta_ref90_lr_{lr}_ws_1.0_its_10_nims_1'
            # save_name = f'tta_adv_lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'

            # load the results for given severity and learning rate
            if not os.path.exists(f'{folder}/{weather_cond}/{save_name}.npy'):
                print(f'File {folder}/{weather_cond}/{save_name} does not exist')
            #     3 x kind x sample x it
            # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
            deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{weather_cond}/{save_name}.npy', allow_pickle=True)
            print(seg_iou_losses.shape)
            lrs_results.append(seg_iou_losses)

        # lr x kind x sample x it
        results = np.array(lrs_results)

        none_base = 100 - results[0, -1, :, 0].mean()
        #  results aggregated over samples and distortions, lr x it
        overall_res = results.mean(axis=(1, 2))
        best_it_per_lr = overall_res.argmin(1)
        best_lr_idx = overall_res[np.arange(best_it_per_lr.shape[0]), best_it_per_lr].argmin()
        base = 100 - overall_res[0, 0]
        best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
        rows += [[weather_cond, base, best,
                  best - base, (best - base) / overall_res[0, 0] * 100]]

    #   now print the results as latex table
    table = pd.DataFrame(rows, columns=columns_names)

    styled_table = table.style. \
        hide(axis="index"). \
        format(precision=2)
    # print(styled_table.to_latex(hrules=True))
    s = styled_table.to_latex(hrules=True)

    print(s)


def per_distortion_table_best_res(folder='cityscapes_results/tta/corruptions', pre='distortion', kinds=[]):
    rows = []
    columns_names = ['corruption', 'level', '$\text{mIoU}_0}$', '$\text{mIoU}_\text{best}$',
                     'diff abs', 'diff cor rel', 'diff cor total', 'best it', 'best lr']
    # severities = [5, 3, 1]
    severities = [3, 2, 1]
    tent_lrs = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4]
    sam_iou_lrs = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    ref_lrs = [5e-6, 1e-5, 5e-5, 1e-4]

    # evaluate reconstruction finetuning only

    freeze_rec = True

    results = []
    lrs = sam_iou_lrs
    for severity in severities:
        lrs_results = []
        for lr in lrs:
            #                 # tta_rec_lr_0.01_freeze_rec_1_its_10_sev_5.npy
            save_name = f'sam_tta_sam-iou_lr_{lr}_ws_1_its_10_nims_1_sev_{severity}'
            # save_name = f'ref_tta_ref90_lr_{lr}_ws_1_its_10_nims_1_sev_{severity}'
            # sam_tta_sam-iou_lr_0.001_ws_1_its_10_nims_1_sev_1.npy

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
    w2s = [0.01, 0.002, 0.001, 0.0002]


    # evaluate reconstruction finetuning only

    freeze_rec = True

    results = []
    for severity in severities:
        w2_results = []
        for w2 in w2s:
            #                 # tta_rec_lr_0.01_freeze_rec_1_its_10_sev_5.npy
            save_name = f'tta_rec&refcorr_lr_{lr}_ws_{w1}_{w2}_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_1_sev_{severity}'

            # load the results for given severity and learning rate
            if not os.path.exists(f'{folder}/{save_name}.npy'):
                print(f'File {save_name} does not exist')
            #     3 x kind x sample x it
            # rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
            deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
            w2_results.append(seg_iou_losses)

        results.append(w2_results)
    # sev x lr x kind x sample x it
    results = np.array(results)

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


def compare_results_freeze_rec(folder='pascal_tta_20_distorted', pre='distortion', kinds=[]):
    rows = []
    columns_names = ['frozen rec. dec.', 'TTA_REC', 'TTA_SEG', 'mIoU']

    severities = [5, 3, 1]
    lrs = [5e-2, 1e-2, 5e-3]
    # evaluate reconstruction finetuning only

    freeze_rec_vals = [True, False]

    for freeze_rec in freeze_rec_vals:
        for i, (rw, sw) in enumerate(([1.0, 0.0], [0.0, 1.0], [1.0, 1.0])):
            row = [freeze_rec, rw, sw]
            results = []
            # aggregate results
            for severity in severities:
                lrs_results = []
                for lr in lrs:
                    save_name = f'lr_{lr}_ws_1_freeze_rec_{int(freeze_rec)}_its_10_rw_{rw}_sw_{sw}_sev_{severity}'

                    # load the results for given severity and learning rate
                    if not os.path.exists(f'{folder}/{save_name}.npy'):
                        print(f'File {save_name} does not exist')
                    #     3 x kind x sample x it
                    rec_mses, seg_deep_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)
                    lrs_results.append(seg_iou_losses)

                results.append(lrs_results)

            # sev x lr x kind x sample x it
            results = np.array(results)

            #  results aggregated over samples and distortions, lr x it
            overall_res = results.mean(axis=(0, 2, 3))
            best_it_per_lr = overall_res.argmin(1)
            best_lr_idx = overall_res[np.arange(3), best_it_per_lr].argmin()
            if i == 0 and freeze_rec:
                base = 100 - overall_res[0, 0]
                rows += [[0, 0, 0, base]]
            best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
            rows += [[freeze_rec, rw, sw, best]]

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
    methods = ['rec', 'ioucorr', 'refcorr', 'tent', 'adv']
    columns_names = ['corruption', 'level', 'base'] + [m for m in methods]

    severities = [5, 3, 1]
    # lrs = [5e-2, 1e-2, 5e-3]
    # high_lrs = [5e-2]
    # low_lrs = [0.0005]
    method_lrs = {'tent': [5e-3], 'rec': [5e-2], 'refcorr': [5e-4], 'ioucorr': [5e-2], 'adv': [0.1]}
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


def analyze_single_run(folder='pascal_tta_clean_20_distorted', kinds=[]):
    methods = ['iou', 'ref', 'tent', 'rec']
    method_lr = {'tent': 5e-3, 'rec': 5e-2, 'ref': 5e-4, 'iou': 5e-4}
    for method in methods:
        lr = method_lr[method]
        severity = 5
        save_name = f'tta_{method}_lr_{lr}_freeze_rec_1_its_10_gradclip_-1_nims_1_sev_{severity}'

        deep_tta_losses, seg_iou_losses = np.load(f'{folder}/{save_name}.npy', allow_pickle=True)

        print()
        #     aggregate kinds x samples x its to (kinds x samples) x its
        seg_iou_losses = seg_iou_losses.reshape(-1, seg_iou_losses.shape[-1]) * 100
        # rel  > 0 means that the method improved the result
        relative_differences = (seg_iou_losses[:, 0] - seg_iou_losses[:, -1]) / seg_iou_losses[:, 0]

        #   in one plot, show the histogram of relative differences, and the mean and std, in another plot, show the number of improved and deteriorated samples
        plt.subplots(1, 2, figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # TODO make sure no bin contains both positive and negative values
        _, _, bars = plt.hist(relative_differences, bins=20)
        for bar in bars:
            if bar.get_x() + bar.get_width() > 0:
                bar.set_facecolor("blue")
            else:
                bar.set_facecolor("red")
        plt.title(f'Distribution of relative improvement')
        plt.xlabel('Relative improvement')
        plt.ylabel('Number of samples')
        # plt.show()

        plt.subplot(1, 2, 2)
        improved = (relative_differences > 0).sum()
        deteriorated = (relative_differences < 0).sum()
        # improved in blue, deteriorated in red
        plt.bar(['improved', 'deteriorated'], [improved, deteriorated], color=['blue', 'red'])
        # plt.bar(['improved', 'deteriorated'], [improved, deteriorated])
        plt.title(f'Number of improved and deteriorated samples')
        plt.ylabel('Number of samples')

        plt.suptitle(f'IoU of {method} over 10 iterations, lr={lr}, severity={severity}')
        plt.show()


# TODO tables: Compare best results for each method side by side
# TODO graphs: Make the tables visual
# TODO tables: plot histogram of improvements/deterioration for each method
# TODO tables:grad clipping for other methods (all but rec needs to be generated)

def get_agg_results_nim(folder='pascal_tta_clean_120_distorted', kinds=[]):
    #     method_lrs = {'tent': [5e-3], 'rec': [5e-2], 'ref': [5e-4], 'iou': [5e-4]} TODO generate results for tent, ref, iou
    method_lrs = {'tent': [5e-2], 'rec': [1e-2], 'ref': [5e-4], 'iou': [5e-4]}

    methods = ['rec']

    rows = []

    columns_names = ['method', '1', '3', '5', '8']
    severities = [5, 3, 1]
    # rec, tent lrs
    # ref, iou lrs
    # lrs = [1e-5, 5e-5, 1e-4, 5e-4]

    # evaluate reconstruction finetuning only

    freeze_rec = True

    for method in methods:
        row = [method]
        lrs = method_lrs[method]
        for nim in [1, 3, 5, 8]:
            results = []
            for severity in severities:
                lrs_results = []
                for lr in lrs:
                    #                 # tta_rec_lr_0.01_freeze_rec_1_its_10_sev_5.npy
                    save_name = f'tta_{method}_lr_{lr}_freeze_rec_{int(freeze_rec)}_its_10_gradclip_-1_nims_{nim}_sev_{severity}'
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
            best = 100 - overall_res[best_lr_idx, best_it_per_lr[best_lr_idx]]
            row += [best]
        rows.append(row)

    #   now print the results as latex table
    table = pd.DataFrame(rows, columns=columns_names)

    styled_table = table.style. \
        hide(axis="index"). \
        format(precision=2)
    # print(styled_table.to_latex(hrules=True))
    s = styled_table.to_latex(hrules=True)

    print(s)



if __name__ == '__main__':
    distortion_keys = ['frost', 'fog', 'gaussian_noise', 'shot_noise', 'spatter', 'defocus_blur', 'glass_blur', 'gaussian_blur', 'brightness', 'contrast', 'none']
    per_distortion_table_best_res(kinds=distortion_keys)
    # per_weather_table_best_res()
    # per_distortion_table_best_res_comb(kinds=distortion_keys)
    # get_agg_results_nim(kinds=distortion_keys)
    # compare_results_ttas_grad_clip(kinds=distortion_keys)
    # compare_results_ttas(kinds=distortion_keys)
    # compare_results_ttas_methods(kinds=distortion_keys)
    # analyze_single_run(kinds=distortion_keys)
    # compare_results_freeze_rec(kinds=distortion_keys)
    # per_distortion_lr_plot(kinds=distortion_keys)
