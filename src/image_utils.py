import cv2


def prepare(img,
            resize=False, new_size=(64, 64),
            apply_contrast=False, contrast_channels=(0, 1, 2)
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
    equalized = img.copy()

    for k in channels:
        equalized[:, :, k] = cv2.equalizeHist(img[:, :, k])

    return equalized

