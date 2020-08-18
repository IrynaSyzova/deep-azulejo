"""
Utility functions to select images suitable for the project.
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def get_square_imgs(img_list, folder = '', tolerance=0.1, plot_aspect_ratios=True):
    """
    Returns list of files that are square or almost square
    
    :param img_list: list of image names 
    :param folder: path to images
    :param tolerance: maximum possible aspect ratio is 1+tolerance
    :param plot_aspect_ratios: if True, plots aspect ratios
    :return: list of square-ish tiles
    """

    aspect_ratios = [0] * len(img_list)

    i = 0

    sample_rectangular = []
    sample_square = []

    for img_file in img_list:
        img = cv2.imread('{}/{}'.format(folder, img_file))[...,::-1]
        aspect_ratios[i] = max(
            img.shape[0]*1.0 / img.shape[1],
            img.shape[1]*1.0 / img.shape[0]
        )
        
        if aspect_ratios[i] >= 1+tolerance:
            sample_rectangular.append(img_file)
            
        else:
            sample_square.append(img_file)
        i += 1

    if i < len(aspect_ratios):
        aspect_ratios = aspect_ratios[:i]
    if plot_aspect_ratios:
        plt.plot(sorted(aspect_ratios));
        plt.gca().axhline(1 + tolerance, color='red')
        title = 'Aspect ratios of the images'
        plt.title(title)
        plt.show()
        
    return sample_square