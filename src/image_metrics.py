"""
Self-defined functions to measure images according to different properties.
"""
import numpy as np
from skimage.metrics import structural_similarity as ssim


def image_aspect_ratio(img):
    """
    Returns aspect ratio of img as measure of squareness
    :param img: image
    :return: aspect ratio, with 0 being least square and 1 being most square
    """
    return min(
        img.shape[0]*1.0 / img.shape[1],
        img.shape[1]*1.0 / img.shape[0]
    )


def tile_uniform_contrast(tile, n_pieces=8):
    """
    Checks contrasts in tile cut into n_pieces x n_pieces, returns average contrast in pieces
    :param tile: tile to check
    :param n_pieces: number of pieces to check individually
    :return: contrast measure, with 0 being least contrast and 1 being most contrast
    """
    channels = (0, 1, 2)
    
    intensity = [tile.img[:, :, k].max() - tile.img[:, :, k].min() for k in channels]
    # max difference per channel

    tile_list = tile.get_pieces(n_pieces)
    
    return np.mean([
        min([
            (tile_piece.img[:, :, k].max() -
             tile_piece.img[:, :, k].min()) * 1.0 / intensity[k] 
            for tile_piece 
            in tile_list
        ])
        for k in channels
    ])


def tile_symmetry(tile, metric=None, agg=None, **kwargs):
    """
    Checks symmetry of tile using provided metric and aggregation function agg.
    I check symmetry of tile itself + it's middle third,
    assuming that middle third is one of the most important parts of the image
    Example:
    ooo
    oxo => x is middle third
    ooo
    :param tile: tile to check
    :param metric: function of 2 images that indicates their similarity
    :param agg: aggregation function
    :param kwargs: args of metric function
    :return: measure of symmetry
    """
    if metric is None:
        def similarity_ssim_based(x, y): return ssim(x, y, multichannel=True)

        metric = similarity_ssim_based
    if agg is None:
        agg = max

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
