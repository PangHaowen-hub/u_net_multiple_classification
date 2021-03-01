import numpy as np
import SimpleITK as sitk
import os

# # 读取.npy文件,每个npy文件中保存两个矩阵，其中第一个CT图像，第二个为mask
arrnpy1 = np.load("preprocessed/lobe_001.npy")  # (2, 234, 204, 204)
# arrnpy2 = np.load("preprocessed/lobe_002.npy")  # (2, 180, 204, 204)
# arrnpy3 = np.load("preprocessed/lobe_003.npy")  # (2, 196, 204, 204)
# arrnpy4 = np.load("preprocessed/lobe_004.npy")  # (2, 200, 204, 204)

# # 读取npz
# arrnpz1 = np.load("preprocessed/stage1/lobe_001.npz")
# arrnpz2 = np.load("preprocessed/stage1/lobe_002.npz")
# arrnpz3 = np.load("preprocessed/stage1/lobe_003.npz")
# arrnpz4 = np.load("preprocessed/stage1/lobe_004.npz")
# x1 = arrnpz1['data']
# x2 = arrnpz2['data']
# x3 = arrnpz3['data']
# x4 = arrnpz4['data']

# npy转为nii
mask_img_arr0 = arrnpy1[0]
new_mask_img0 = sitk.GetImageFromArray(mask_img_arr0)
mask_img_arr1 = arrnpy1[1]
new_mask_img1 = sitk.GetImageFromArray(mask_img_arr1)
sitk.WriteImage(new_mask_img0, 'npy10.nii.gz')
sitk.WriteImage(new_mask_img1, 'npy11.nii.gz')

# # 生成测试数据
# x0 = np.random.rand(2, 128, 128, 128)
# x1 = np.random.rand(2, 128, 128, 128)
# x2 = np.random.rand(2, 128, 128, 128)
# x3 = np.random.rand(2, 128, 128, 128)
# x4 = np.random.rand(2, 128, 128, 128)
# np.save('x0.npy', x0)
# np.save('x1.npy', x1)
# np.save('x2.npy', x2)
# np.save('x3.npy', x3)
# np.save('x4.npy', x4)

print("load .npy done")

