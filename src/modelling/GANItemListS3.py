import PIL
import warnings
import numpy as np

from fastai.vision.gan import GANItemList
from fastai.vision.image import Image, pil2tensor

from src import s3_utils


class GANItemListS3(GANItemList):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open(self, s3_path):
        return __open_image(s3_path, convert_mode=self.convert_mode, after_open=self.after_open)

    @classmethod
    def from_s3(cls, s3_path, **kwargs):
        return cls(s3_utils.get_image_list_from_s3(s3_path), path='', **kwargs)


def __open_image(s3_path, convert_mode='RGB', cls=Image, after_open=None):
    "Return `Image` object created from image in file `fn`."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(s3_path).convert(convert_mode)
    if after_open: x = after_open(x)
    x = pil2tensor(x,np.float32)
    return cls(x)
