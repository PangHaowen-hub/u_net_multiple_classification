import numpy as np
import SimpleITK as sitk
import os

# # 读取.npy文件,每个npy文件中保存两个矩阵，其中第一个CT图像，第二个为mask
# arrnpy0 = np.load("preprocessed/lobe_000.npy")  # (2, 198, 204, 204)
# arrnpy1 = np.load("preprocessed/lobe_001.npy")  # (2, 234, 204, 204)
# arrnpy2 = np.load("preprocessed/lobe_002.npy")  # (2, 180, 204, 204)
# arrnpy3 = np.load("preprocessed/lobe_003.npy")  # (2, 196, 204, 204)
# arrnpy4 = np.load("preprocessed/lobe_004.npy")  # (2, 200, 204, 204)

# # 读取npz
# arrnpz0 = np.load("preprocessed/stage1/lobe_000.npz")
# arrnpz1 = np.load("preprocessed/stage1/lobe_001.npz")
# arrnpz2 = np.load("preprocessed/stage1/lobe_002.npz")
# arrnpz3 = np.load("preprocessed/stage1/lobe_003.npz")
# arrnpz4 = np.load("preprocessed/stage1/lobe_004.npz")
# x0 = arrnpz0['data']
# x1 = arrnpz1['data']
# x2 = arrnpz2['data']
# x3 = arrnpz3['data']
# x4 = arrnpz4['data']

# # npy转为nii
# mask_img_arr0 = arrnpy0[0]
# new_mask_img0 = sitk.GetImageFromArray(mask_img_arr0)
# mask_img_arr1 = arrnpy0[1]
# new_mask_img1 = sitk.GetImageFromArray(mask_img_arr1)
# sitk.WriteImage(new_mask_img0, 'npy0.nii.gz')
# sitk.WriteImage(new_mask_img1, 'npy1.nii.gz')

print("load .npy done")

