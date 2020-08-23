import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def crop(img):
    new_shape = min(img.shape[0], img.shape[1])
    
    return img[0:new_shape, 0:new_shape, ...]


def resize(img, new_size=(64, 64)):
    return cv2.resize(img, new_size)
    

def increase_contrast(img):
    equalized = [cv2.equalizeHist(img[:, :, k]) for k in (0, 1, 2)]

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
