import torch.utils.data as data
from torchvision.transforms import transforms
import numpy as np
from os.path import splitext
from os import listdir
import logging
from torch.utils.data import DataLoader
import SimpleITK as sitk


class MyDataset(data.Dataset):
    # def __init__(self, imgs_dir, masks_dir, mask_suffix=''):
    def __init__(self, imgs_dir):
        self.imgs_dir = imgs_dir
        # self.masks_dir = masks_dir
        # self.mask_suffix = mask_suffix
        # splitext分离文件名和扩展名
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith('.')]
        # self.images_transform = transforms.ToTensor()
        # self.masks_transform = transforms.ToTensor()
        self.images_transform = None
        self.masks_transform = None
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = self.imgs_dir + '/' + idx + '.npy'

        npy = np.load(img_file)
        img = npy[0]
        mask = npy[1]
        img = np.expand_dims(img, 0)
        mask = np.expand_dims(mask, 0)
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # one_hot编码
        class_num = 6
        temp_mask = np.zeros((class_num, mask.shape[1], mask.shape[2], mask.shape[3]))  # one_hot编码类别数6
        for i in range(class_num):
            s0 = mask == i
            s0 = s0.squeeze()
            temp_mask[(i, s0)] = 1
        # for j in range(6):
        #     new_mask_img = sitk.GetImageFromArray(temp_mask[j])
        #     sitk.WriteImage(new_mask_img, str(j) + '.nii.gz')

        return img, temp_mask

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    train_dataset = MyDataset("data/train/imgs")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    x = train_dataloader.dataset[0]
    print(train_dataloader)
