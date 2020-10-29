from __future__ import print_function, division
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from src import s3_utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TileDataset(Dataset):
    def __init__(self, s3_key, transform):
        """

        :param root_dir: directory with the images
        """
        self.key = s3_key
        self.pics = s3_utils.get_image_list_from_s3(s3_key)
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        images = [s3_utils.get_image_list_from_s3(key) for key in self.pics[idx]]

        return images


class Recolour(object):
    def __call__(self, img):
        img_result = img.copy()
        channels = (0, 1, 2)
        channels_permuted = np.random.choise(channels, size=3, replace=True)
        for i in range(len(channels)):
            img_result[..., channels[i]] = img[..., channels_permuted[i]]
        return img_result
