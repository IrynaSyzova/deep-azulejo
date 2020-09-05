"""
Self-defined functions to measure images according to different properties.
"""
import cv2
import numpy as np

from src import image_utils
from src.Tile import Tile
from src.utils import _prepare_img


def image_aspect_ratio(img):
    return min(
        img.shape[0]*1.0 / img.shape[1],
        img.shape[1]*1.0 / img.shape[0]
    )


def tile_uniform_contrast(tile, n_pieces=64):
    channels = (0, 1, 2)
    
    intensity = [tile.img[:, :, k].max() - tile.img[:, :, k].min() for k in channels]

    tile_list = tile.get_pieces(n_pieces)
    
    return np.mean([
        min([
            (tile_piece.img[:, :, k].max() - \
             tile_piece.img[:, :, k].min()) * 1.0 / intensity[k] 
            for tile_piece 
            in tile_list
        ])
        for k in channels
    ])


def tile_symmetry(tile, metric, agg, **kwargs):
    tile_center = tile.get_square_from_center(0.33)
    
    tile_compare = [
        [
            tile, 
            tile.flip_transpose()
        ], [
            tile, 
            tile.rotate(clockwise=False).flip_transpose().rotate(clockwise=True)
        ], [
            tile_center, 
            tile_center.flip_transpose()
        ], [
            tile_center,
            tile_center.rotate(clockwise=False).flip_transpose().rotate(clockwise=True)
        ]
    ]

    symmetry_measure_list = []
    for i in range(len(tile_compare)):
        tile0, tile1 = tile_compare[i]
        symmetry_measure = metric(tile0.img, tile1.img, **kwargs)
        symmetry_measure_list.append(symmetry_measure)

    return agg(symmetry_measure_list)
