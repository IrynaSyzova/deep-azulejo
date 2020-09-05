"""
Self-defined functions to measure images according to different properties.
"""
import cv2
import numpy as np
<<<<<<< HEAD
import warnings
=======
>>>>>>> f83ec63b10b0eda1d9a4868b44694179e4aa01db

from src import image_utils
from src.Tile import Tile
from src.utils import _prepare_img


<<<<<<< HEAD
def contrast_measure(tile, agg_by_channel=np.median):
    """
    Defines normalised contrast of the image
    :param img: colour image
    :param agg: aggregation by channel
    :return: contrast measure from [0, 1]
    """
    return agg_by_channel([
        tile.img[:, :, i].max() - tile.img[:, :, i].min()
        for i in [0, 1, 2]
    ])/255.0


def get_tile_contrast_by_multitile(tile, n_pieces=25, agg_by_channel=np.median, agg_subtiles=min):
    """
    Given object of class Tile return minimum contrast of object's image
        cropped into n_pieces parts
    :param tile: object of class Tile
    :param n_pieces: number of pieces to check contrast in
    :param agg_by_channel: how to aggregate contrast by 3 color channels
    :param agg_subtiles: how to aggregate contrast in subtiles
    :return: contrast measure from [0, 1]
    """
=======
def image_aspect_ratio(img):
    return min(
        img.shape[0]*1.0 / img.shape[1],
        img.shape[1]*1.0 / img.shape[0]
    )


def tile_uniform_contrast(tile, n_pieces=64):
    channels = (0, 1, 2)
    
    intensity = [tile.img[:, :, k].max() - tile.img[:, :, k].min() for k in channels]
>>>>>>> f83ec63b10b0eda1d9a4868b44694179e4aa01db

    tile_list = tile.get_pieces(n_pieces)
<<<<<<< HEAD
    return agg_subtiles([contrast_measure(_.img, agg_by_channel) for _ in tile_list])


def get_tile_symmetry(tile, metric, agg, *args, **kwargs):
    """
    Check tile's symmetry horizontally, vertically, and diagonally; returns best of possible
    :param tile: object of class Tile
    :param metric: metric to apply
    :param agg: method of aggregation, for example, min, max, np.median
    :param args: additional arguments for metric
    :param kwargs: additional keyword arguments for metric
    :return: symmetric measure of tile
    """
=======
    
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
    
>>>>>>> f83ec63b10b0eda1d9a4868b44694179e4aa01db
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
<<<<<<< HEAD
        symmetry_measure = metric(tile0.img, tile1.img, *args, **kwargs)
=======
        symmetry_measure = metric(tile0.img, tile1.img, **kwargs)
>>>>>>> f83ec63b10b0eda1d9a4868b44694179e4aa01db
        symmetry_measure_list.append(symmetry_measure)

    return agg(symmetry_measure_list)

<<<<<<< HEAD

def get_symmetry_by_multitile(tile, n_pieces, agg_pieces, metric, agg_metric, *args, **kwargs):
    """
    Checks tile symmetry separately for each of n_pieces sub-tiles
    :param tile: object of class Tile
    :param n_pieces: number of pieces to chop tile into
    :param agg_pieces: how to aggregate symmetry measures of sub-tiles
    :param metric: symmetry metric
    :param agg_metric: method of aggregation
    :param args: additional arguments for metric
    :param kwargs: additional keyword arguments for metric
    :return: symmetric measure of tile
    """
    if ((n_pieces) ** 0.5) ** 2 != n_pieces:
        warnings.warn('{} should be a square number.'.format(n_pieces))
        return None
    tile_list = tile.get_pieces(n_pieces)
    return agg_pieces([get_tile_symmetry(tile_piece, metric, agg_metric, *args, **kwargs) for tile_piece in tile_list])



def get_multitileness(tile, pieces_to_check, agg_final, agg_pieces, metric, agg_metric, *args, **kwargs):
    """
    TODO
    """
    symmetry_pieces = []
    for n in pieces_to_check:
        symmetry_pieces.append(get_symmetry_by_multitile(tile, n, agg_pieces, metric, agg_metric, *args, **kwargs))

    return agg_final(symmetry_pieces)
=======
        
>>>>>>> f83ec63b10b0eda1d9a4868b44694179e4aa01db
