"""
Utility functions to select images suitable for the project.
"""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from src import tile_metrics
from src.Tile import Tile
import warnings


def select_imgs(img_list, folder, func, **kwargs):
    """

    :param img_list:
    :param folder:
    :param func:
    :param kwargs:
    :return:
    """
    pass


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

    for i, img_file in enumerate(img_list):
        img = cv2.imread('{}/{}'.format(folder, img_file))[...,::-1]
        aspect_ratios[i] = max(
            img.shape[0]*1.0 / img.shape[1],
            img.shape[1]*1.0 / img.shape[0]
        )
        
        if aspect_ratios[i] >= 1+tolerance:
            sample_square.append(img_file)

    if plot_aspect_ratios:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        ax.plot(sorted(aspect_ratios))
        ax.axhline(1 + tolerance, color='red')
        title = 'Aspect ratios of the images'
        ax.set_title(title)
        plt.show()
        
    return sample_square


def get_contrast_imgs(img_list, folder = '', min_contrast=0.25, n_pieces=25, plot_contrasts=True):
    """
    Returns list of files that are high contrast
    
    :param img_list: list of image names 
    :param folder: path to images
    :param min_contrast:minimum possible contast measure as defined in tile_metrics.py
    :param plot_contrasts: if True, plots aspect ratios
    :return: list of contasting enough files
    """
    contrast_measure_list = [0] * len(img_list)
    sample_contrasting = []

    for i, img_file in enumerate(img_list):
        img = cv2.imread('{}/{}'.format(folder, img_file))[...,::-1]
        img = _prepare_img(img)

        contrast_measure_list[i] =  tile_metrics.get_tile_contrast(Tile(img), n_pieces=25)
        if contrast_measure_list[i] > min_contrast:
            sample_contrasting.append(img_file)

    if plot_contrasts:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax.plot(sorted(contrast_measure_list))
        ax.axhline(min_contrast, color='red')
        title = 'Contrast measures of the images'
        ax.set_gtitle(title)
        plt.show()
    
    return sample_contrasting


def get_symmetric_imgs(img_list, folder = '', cut_off_mse=0.05, plot_symmetry_measures=True):
    """
    Returns list of files that are symmetric using normalized mse and structual similarity from skimage
    
    :param img_list: list of image names 
    :param folder: path to images
    :param cut_offs: minimum symmetry measure by metric as defined in tile_metrics.py
    :param plot_symmetry_measures: if True, plots symmetry measures
    :return: list of symmetric enough files
    """

    symmetry_measure = [None] * len(img_list)
    
    sample_symmetric = []
    
    for i, img_file in enumerate(img_list):
        img = cv2.imread('{}/{}'.format(folder, img_file))[...,::-1]
    
        symmetry_measure[i] = tile_metrics.get_tile_symmetry(Tile(_prepare_img(img)))

        if (symmetry_measure[i]['ssim'] >= cut_off_ssim) or \
            (symmetry_measure[i]['normalized_root_mse'] <= cut_off_mse):
            sample_symmetric.append(img_file)

    if plot_symmetry_measures:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(sorted([_['normalized_root_mse'] for _ in symmetry_measure]))
        ax[0].axhline(cut_off_mse, color='red')
        title = 'Symmetry measures of the images (mse)'
        ax[0].set_title(title)
        
        ax[1].plot(sorted([_['ssim'] for _ in symmetry_measure]))
        ax[1].axhline(cut_off_ssim, color='red')
        title = 'Symmetry measures of the images (ssim)'
        ax[1].set_title(title)
        plt.show()
    
    return sample_symmetric


def get_multitile_imgs(img_list, folder='', cut_off_mse=0.05, cut_off_ssim=0.95, plot_symmetry_measures=True):
    """
    Returns list of files that are likely to consist of multiple symmetric images
    using normalized mse and structual similarity from skimage

    :param img_list: list of image names
    :param folder: path to images
    :param cut_offs: minimum symmetry measure by metric as defined in tile_metrics.py
    :param plot_symmetry_measures: if True, plots symmetry measures
    :return: list of images consisting of multiple symmetric tiles
    """

    symmetry_pieces = [None] * len(img_list)

    sample_multitile = []

    for i, img_file in enumerate(img_list):
        img = cv2.imread('{}/{}'.format(folder, img_file))[..., ::-1]

        symmetry_pieces[i] = tile_metrics.get_tile_symmetry_by_piece(Tile(_prepare_img(img, new_size=(512, 512))))

        if (symmetry_pieces[i]['ssim'] >= cut_off_ssim) or \
                (symmetry_pieces[i]['normalized_root_mse'] <= cut_off_mse):
            sample_multitile.append(img_file)

    if plot_symmetry_measures:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(sorted([_['normalized_root_mse'] for _ in symmetry_pieces]))
        ax[0].axhline(cut_off_mse, color='red')
        title = 'Symmetry measures of the images by pieces (mse)'
        ax[0].set_title(title)

        ax[1].plot(sorted([_['ssim'] for _ in symmetry_pieces]))
        ax[1].axhline(cut_off_ssim, color='red')
        title = 'Symmetry measures of the images by pieces (ssim)'
        ax[1].set_title(title)
        plt.show()

    return sample_multitile


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
    