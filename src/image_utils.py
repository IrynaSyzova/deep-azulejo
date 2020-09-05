import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def prepare(img, 
            resize=False, new_size=(64, 64), 
            apply_contrast=False, contrast_channels=(0, 1, 2), 
            apply_blur=False, blur_kernel=(3, 3)
           ):
    new_img = crop(img)
    
    if resize:
        new_img = cv2.resize(new_img, new_size)
    
    if apply_contrast:
        new_img = increase_contrast(new_img, channels=contrast_channels)
    
    if apply_blur:
        new_img = blur(new_img, blur_kernel)
    
    return new_img


def crop(img):
    new_shape = min(img.shape[0], img.shape[1])
    
    return img[0:new_shape, 0:new_shape, ...]


def resize(img, new_size=(256, 256)):
    return cv2.resize(img, new_size)
    

def increase_contrast(img, channels=(0, 1, 2)):
    equalized = [cv2.equalizeHist(img[:, :, k]) if k in channels else img[:, :, k] for k in (0, 1, 2)]

    return np.array(
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


def blur(img, kernel=(5, 5)):
    return cv2.GaussianBlur(img, kernel, 0)
