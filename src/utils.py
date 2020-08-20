import cv2


def _prepare_img(img, new_size=(128, 128), blurring_kernel=(3, 3)):
    """
    Reshapes and blurs image in preparation for contrast and symmetry checks.
    :param img: image to prepare
    :return: prepared image
    """
    new_shape = min(img.shape[0], img.shape[1])
    img = img[0:new_shape, 0:new_shape, ...]

    img = cv2.resize(img, new_size)
    img = cv2.GaussianBlur(img, blurring_kernel, 0)

    return img