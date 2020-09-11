import cv2
import numpy as np


def prepare(img,
            resize=False, new_size=(64, 64),
            apply_contrast=False, contrast_channels=(0, 1, 2),
            apply_blur=False, blur_kernel=(3, 3)
           ):
    """
    Function to prepare the file by cropping, resizing, increasing contrast, and blurring
    :param img: image to prepare
    :param resize: boolean, to resize or not
    :param new_size: if resize, what new size will be
    :param apply_contrast: boolean, to increase contrast or not
    :param contrast_channels: if apply contrast, in what colour channels
    :return: prepared image
    """
    new_img = crop(img)

    if resize:
        new_img = cv2.resize(new_img, new_size)

    if apply_contrast:
        new_img = increase_contrast(new_img, channels=contrast_channels)

    return new_img


def crop(img):
    """
    Crops image to square starting from upper left
    :param img: image to crop
    :return: cropped image
    """
    new_shape = min(img.shape[0], img.shape[1])
    
    return img[0:new_shape, 0:new_shape, ...]


def resize(img, new_size=(256, 256)):
    """
    Resizes image
    :param img: image to resize
    :param new_size: size after resizing
    :return: resized image
    """
    return cv2.resize(img, new_size)
    

def increase_contrast(img, channels=(0, 1, 2)):
    """
    Increases contrast in an image
    :param img: image to alter
    :param channels: list of channels to increase contrast in
    :return: altered image
    """
    equalized = [cv2.equalizeHist(img[:, :, k]) for k in (0, 1, 2)]
    equalized_img = np.array(
        [
            [
                [
                    equalized[k][i, j]
                    for k
                    in (0, 1, 2)
                ]
                for j in range(img.shape[0])
            ]
            for i in range(img.shape[1])
        ]
    )

    return assemble_img_by_channel(equalized_img, img, channels)


def assemble_img_by_channel(img1, img2, channels):
    """
    Assembles a new image by using layers from img1 and img2;
    for colour channels in channels we use img1, otherwise img2;
    E.g.: for channels = [0, 1, 2] result will be img1, for channels = [], result will be img2
    :param img1: starting image
    :param img2: another starting image
    :param channels: colour channels to take from image 1
    :return: new image
    """
    result = [img1[:, :, k] if k in channels else img2[:, :, k] for k in (0, 1, 2)]
    return np.array(
        [
            [
                [
                    result[k][i, j]
                    for k
                    in (0, 1, 2)
                ]
            for j in range(img1.shape[0])
            ]
            for i in range(img1.shape[1])
        ]
    )
