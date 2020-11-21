from __future__ import print_function, division
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from tenacity import retry, wait_fixed, stop_after_attempt

from src import s3_utils
from src.logging_utils import get_logger

logger = get_logger('dataset')


class TileDataset(Dataset):
    def __init__(self, s3_key, transform = None):
        """
        :param s3_key: s3 path with the images
        :param transform: transformation for the images
        """
        # super(TileDataset, self).__init__()
        self.key = s3_key
        self.pics = s3_utils.get_image_list_from_s3(s3_key)
        self.transform = transform

    def __len__(self):
        return len(self.pics)

    def __getitem__(self, idx):
        try:
            return self.__get_image(idx)
        except Exception as e:
            logger.warning('Problem with __getitem__ for item {}: {}'.format(self.pics[idx], e))
            return None

    @retry(wait=wait_fixed(0.02), stop=stop_after_attempt(3))
    def __get_image(self, idx):
        image = s3_utils.read_image_from_s3(self.pics[idx], as_array=False)

        if self.transform:
            image = self.transform(image)

        return image


def collate_fn(batch):
    new_batch = [img for img in batch if img is not None]
    if len(new_batch) < len(batch):
        logger.warning('Collate function filtered {} images out of {}'.format(len(batch) - len(new_batch), len(batch)))
    return default_collate(list(new_batch))
