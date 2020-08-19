"""
Self-defined functions to measure images according to different properties.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, normalized_root_mse
import warnings

from src.Tile import Tile


def _contrast_measure(img):
    """
    Defines normalised contrast of the image
    :param img: colour image
    :return: contrast measure from [0, 1]
    """
    return np.median([img[:, :, i].max() - img[:, :, i].min() for i in [0, 1, 2]])/255.0


def get_tile_contrast(tile, n_pieces=25):
    """
    Given object of class Tile return minimum contrast of object's image cropped into n_pieces parts
    :param tile: object of class Tile
    :param n_pieces: number of pieces to check contrast in
    :return: contrast measure from [0, 1]
    """

    if ((n_pieces)**0.5)**2 != n_pieces:
        warnings.warn('{} should be a square number.'.format(n_pieces))
    tile_list = tile.get_pieces(n_pieces)
    return min([contrast_measure(_.img) for _ in tile_list])


def get_tile_symmetry(tile, metric='structural_similarity'):
    """
    Checks tile's symmetry horizontally, vertically, and diagonally; returns best of them.
    :param tile: object of class Tile
    :param metric: metric to check; 
        current possible values are `structural_similarity` or `normalized_root_mse`
    :return: measure of tiles to check contrast in
    """
    if metric not in ('structural_similarity', 'normalized_root_mse'):
        warnings.warn('{} is not currently supported'.format(metric))
        return None
    
    tile_compare = [
        [tile, tile.flip_horizontal()],
        [tile, tile.flip_vertical()],
        [tile, tile.flip_transpose()],
        [tile, tile.rotate(clockwise=False).flip_transpose().rotate(clockwise=True)]
    ]

    symmetry_measure_list = []
    for i in range(len(tile_compare)):
        tile0, tile1 = tile_compare[i]
        if metric == 'structural_similarity':
            symmetry_measure = ssim(tile0.img, tile1.img, multichannel=True)
        elif metric == 'normalized_root_mse':
            symmetry_measure = normalized_root_mse(tile0.img, tile1.img)
        else:
            warnings.warn('{} is not currently supported'.format(metric))
            return None
        symmetry_measure_list.append(symmetry_measure)

    if metric == 'structural_similarity':
        return np.max(symmetry_measure_list)
    if metric == 'normalized_root_mse':
        return np.min(symmetry_measure_list)


def get_tile_symmetry_by_pieces(tile, pieces=(4, 9, 16, 25, 36, 49, 64), metric='structural_similarity'):
    """
    Checks tile symmetry
    :param tile: object of class Tile
    :param pieces: list of numbers of pieces to symmetry in
    :return: best symmetry obtained
    """
    symmetry_pieces = []
    for n in pieces:
        if ((n) ** 0.5) ** 2 != n:
            warnings.warn('{} should be a square number.'.format(n))
            continue

        tile_list = tile.get_pieces(n)
        
        if metric == 'structural_similarity':
            symmetry_by_piece = np.min([
                get_tile_symmetry(tile_piece, metric='structural_similarity')
                for tile_piece 
                in tile_list
            ])
            symmetry_pieces.append(symmetry_by_piece)
        elif metric = 'normalized_root_mse':
            symmetry_by_piece = np.max([
                get_tile_symmetry(tile_piece, metric='normalized_root_mse') 
                for tile_piece 
                in tile_list
            ])
            symmetry_pieces.append(symmetry_by_piece)
    
    if 'structural_similarity':
        return max(symmetry_pieces)
    if 'normalized_root_mse': 
        return min(symmetry_pieces)
    