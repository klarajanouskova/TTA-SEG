from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from eval_segmentation import load_sam
import torch
import cv2


def adversarial_attack(img_path='../animals/medove.jpg'):
    img = np.array(Image.open(img_path))

    # resize img  by 1/5
    img = cv2.resize(img, (img.shape[1]//5, img.shape[0]//5))

    mask_generator = load_sam('sam_vit_b.pth')

    panda_pts = np.array([[[0.35, 0.4]]])
    tiger_pts = np.array([[[0.5, 0.5]]])
    giraffe_pts = np.array([[[0.4, 0.4]]])
    meda_pts = np.array([[[0.5, 0.5]]])
    medove_pts = np.array([[[0.4, 0.8]]])
    pts = medove_pts
    # scale pts by image size
    pts = pts * np.array([[img.shape[1], img.shape[0]]])
    labels = np.array([[1]])

    pts = torch.as_tensor(pts, device=mask_generator.predictor.device)
    labels = torch.tensor(labels, dtype=torch.int, device=pts.device)

    adv_img = mask_generator.pgd_attack(img, pts, labels, gt='invert', debug=True, iters=10, lr=0.0001)

    # plt.imshow(img)
    # plt.show()

if __name__ == '__main__':
    adversarial_attack()