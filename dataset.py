import torch.utils.data as data
import os
import PIL.Image as Image
import SimpleITK as sitk
from torchvision.transforms import transforms
import numpy as np
from os.path import splitext
from os import listdir
import logging
import glob
from torch.utils.data import DataLoader


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

        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        # mask = np.array(mask)
        # mask = np.expand_dims(mask, axis=2)
        # semantic_map = []
        # palette = [[0], [42], [85], [127], [170], [255]]
        # for colour in palette:
        #     equality = np.equal(mask, colour)
        #     class_map = np.all(equality, axis=-1)
        #     semantic_map.append(class_map)
        # mask = np.stack(semantic_map, axis=-1)
        # img = self.images_transform(img)
        # mask = self.masks_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    train_dataset = MyDataset("data/train/imgs")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print(train_dataloader)
