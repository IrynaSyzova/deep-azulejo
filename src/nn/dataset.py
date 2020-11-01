from __future__ import print_function, division
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
        super(TileDataset, self).__init__()
        self.key = s3_key
        self.pics = s3_utils.get_image_list_from_s3(s3_key)
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        image = Image.fromarray(s3_utils.read_image_from_s3(self.pics[idx]))

        if self.transform:
            image = self.transform(image)

        return image
