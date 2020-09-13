import numpy as np
import uuid
import cv2

from collections import deque
from itertools import product, combinations

from src import Tile
from src import image_utils

import logging

logger = logging.getLogger(__name__)


def enrich(tile, save_func, scale_min=0.25, scale_max=4):
    """
    Creates a list of new tiles obtained by cutting and gluing current tile.

    :param tile: starting tile
    :param save_func: function to write generated images; should be called as save_func(image, image_path)
    :param scale_min: how small minimum size of obtained tile can be relative to starting tile by dimension
    :param scale_max: how big maximum size of obtained tile can be relative to starting tile by dimension
    :return: None
    """
    logger.info('Enriching tile of {} dims'.format(tile.dims))

    min_size = int(tile.dims[0] * scale_min)
    max_size = tile.dims[0] // scale_max

    def __save_tile(tile_save):
        logger.info('Saving tile of {} dims.'.format(tile_save.dims))
        for x in enrich_colour(tile_save):
            img_name = str(uuid.uuid4())
            save_func(x.img, '{}.jpg'.format(img_name))
        logger.info('Saving finished.')

    to_fragment = deque()
    to_augment = deque()

    to_fragment.append(tile)

    while to_fragment:
        curr = to_fragment.popleft()

        __save_tile(curr)

        rhombus = curr.get_rhombus()
        if rhombus.dims[0] >= min_size:
            to_fragment.append(rhombus)
            to_augment.append(rhombus)
            __save_tile(rhombus)

        no_center = curr.remove_center()
        if no_center.dims[0] >= min_size:
            to_fragment.append(no_center)
            to_augment.append(no_center)
            __save_tile(no_center)

        border_constant = curr.add_border(border_thickness=0.05, border_type=cv2.BORDER_REPLICATE)
        to_augment.append(border_constant)
        __save_tile(border_constant)

        border_reflect = curr.add_border(border_thickness=0.15, border_type=cv2.BORDER_REFLECT)
        to_fragment.append(border_reflect)
        to_augment.append(border_reflect)
        __save_tile(border_reflect)

        if curr.get_quadrant(0, 0).dims[0] >= min_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    __save_tile(curr.get_quadrant(0, 1))

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(curr)

        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))

    while to_augment:
        curr = to_augment.pop()
        rhombus = curr.get_rhombus()
        to_augment.append(rhombus)
        __save_tile(rhombus)

        if curr.dims[0] * 4 <= max_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    unfolded = curr.assemble_quadrant_unfold(i, j)
                    to_augment.append(unfolded)
                    __save_tile(unfolded)

            windmill = curr.assemble_quadrant_windmill()
            to_augment.append(windmill)
            __save_tile(windmill)


def enrich_colour(tile):
    """
    Returns a list of tiles obtained from starting tile by recolouring it.

    :param tile: starting tile
    :return: list of recoloured tiles
    """
    logger.info('Recolouring tile')
    img = tile.img
    channels = product([0, 1, 2], repeat=3)
    return [Tile.Tile(np.array([img[..., _[0]], img[..., _[1]], img[..., _[2]]]).transpose())
            for _ in channels
            if len(set(_)) > 1]
