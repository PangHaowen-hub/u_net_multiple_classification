import torch.utils.data as data
import os
import PIL.Image as Image
from torchvision.transforms import transforms
import numpy as np


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if (os.path.splitext(file)[1] == '.png'):
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


class MyDataset(data.Dataset):
    def __init__(self, root_imgs, root_masks):
        imgs = []
        img = get_listdir(root_imgs)
        img.sort()
        mask = get_listdir(root_masks)
        mask.sort()
        n = len(img)
        for i in range(n):
            imgs.append([img[i], mask[i]])
        self.imgs = imgs
        self.images_transform = transforms.ToTensor()
        self.masks_transform = transforms.ToTensor()

    def __getitem__(self, index):
        images_path, masks_path = self.imgs[index]
        image = Image.open(images_path)
        mask = Image.open(masks_path)
        mask = np.array(mask)
        mask = np.expand_dims(mask, axis=2)
        semantic_map = []
        palette = [[0], [42], [85], [127], [170], [255]]
        for colour in palette:
            equality = np.equal(mask, colour)
            class_map = np.all(equality, axis=-1)
            semantic_map.append(class_map)
        mask = np.stack(semantic_map, axis=-1).astype(np.float32)
        image = self.images_transform(image)
        mask = self.masks_transform(mask)
        return image, mask, images_path, masks_path

    def __len__(self):
        return len(self.imgs)


class TestDataset(data.Dataset):
    def __init__(self, root_imgs):
        imgs = []
        img = get_listdir(root_imgs)
        n = len(img)
        for i in range(n):
            imgs.append(img[i])
        self.imgs = imgs
        self.images_transform = transforms.ToTensor()

    def __getitem__(self, index):
        images_path = self.imgs[index]
        image = Image.open(images_path)
        image = self.images_transform(image)
        return image, images_path

    def __len__(self):
        return len(self.imgs)
