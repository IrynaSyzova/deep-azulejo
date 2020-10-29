from __future__ import print_function, division
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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

        images = [Image.fromarray(s3_utils.get_image_list_from_s3(key)) for key in self.pics[idx]]

        if self.transform:
            images = self.transform(images)

        return images
