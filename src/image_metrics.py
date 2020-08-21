"""
Self-defined functions to measure images according to different properties.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, normalized_root_mse

from src.Tile import Tile


def image_aspect_ratio(img):
    return min(
        img.shape[0]*1.0 / img.shape[1],
        img.shape[1]*1.0 / img.shape[0]
    )


def tile_uniform_contrast(tile, n_pieces=16):
    max_intensity = [tile.img[:, :, k].max() for k in (0, 1, 2)]
    min_intensity = [tile.img[:, :, k].min() for k in (0, 1, 2)]
    
    tile_list = tile.get_pieces(n_pieces)
    
    return np.mean([
        np.median([
            (tile_piece.img[:, :, k].max() - tile_piece.img[:, :, k].min())*1.0 /
            (max_intensity[k] - min_intensity[k]) for k in (0, 1, 2)
        ])
        for tile_piece 
        in tile_list
    ])


def _tile_symmetry_helper(tile, metric='ssim'):
    if metric not in ('ssim', 'normalized_root_mse'):
        print('{} is not currently supported'.format(metric))
        return
    
    tile_compare = [
        [tile, tile.flip_horizontal()],
        [tile, tile.flip_vertical()],
        [tile, tile.flip_transpose()],
        [tile, tile.rotate(clockwise=False).flip_transpose().rotate(clockwise=True)]
    ]

    symmetry_measure_list = []
    for i in range(len(tile_compare)):
        tile0, tile1 = tile_compare[i]
        if metric == 'ssim':
            symmetry_measure = ssim(tile0.img, tile1.img, multichannel=True)
        elif metric == 'normalized_root_mse':
            symmetry_measure = normalized_root_mse(tile0.img, tile1.img)
        symmetry_measure_list.append(symmetry_measure)

    if metric == 'ssim':
        return np.max(symmetry_measure_list)
    if metric == 'normalized_root_mse':
        return np.min(symmetry_measure_list)

    
def tile_symmetry(tile):
    symmetry_measure = {}
    
    symmetry_measure['ssim'] = _tile_symmetry_helper(tile, metric='ssim')
    symmetry_measure['normalized_root_mse'] = _tile_symmetry_helper(tile, metric='normalized_root_mse')
    
    return symmetry_measure


def get_tile_symmetry_by_piece(tile, pieces=(4, 9, 16, 25, 36, 49, 64)):
    symmetry_pieces = {'ssim': [], 'normalized_root_mse': []}
    for n in pieces:
        tile_list = tile.get_pieces(n)
        
        symmetry_by_piece = np.min([_tile_symmetry_helper(tile_piece, metric='ssim') for tile_piece in tile_list])
        symmetry_pieces['ssim'].append(symmetry_by_piece)
            
        symmetry_by_piece = np.max([_tile_symmetry_helper(tile_piece, metric='normalized_root_mse') for tile_piece in tile_list])
        symmetry_pieces['normalized_root_mse'].append(symmetry_by_piece)
        
    return {'ssim': max(symmetry_pieces['ssim']), 'normalized_root_mse': min(symmetry_pieces['normalized_root_mse']) }
    