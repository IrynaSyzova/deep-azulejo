import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def prepare(img, new_size=(64, 64)):
    return cv2.resize(crop(img), new_size)


def crop(img):
    new_shape = min(img.shape[0], img.shape[1])
    
    return img[0:new_shape, 0:new_shape, ...]


def resize(img, new_size=(256, 256)):
    return cv2.resize(img, new_size)
    

def increase_contrast(img, channels=None):
    if channels is None:
        channels = (0, 1, 2)
    
    equalized = [cv2.equalizeHist(img[:, :, k]) if k in channels else img[:, :, k] for k in (0, 1, 2)]

    return np.array(
        [
            [
                [
                    equalized[k][i, j]
                     for k
                     in (0, 1, 2)
                ] 
            for i in range(img.shape[0]) 
            ]
            for j in range(img.shape[1])
        ]
    )
