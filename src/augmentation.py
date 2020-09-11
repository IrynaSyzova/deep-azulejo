import numpy as np
import uuid

from collections import deque
from itertools import product, combinations

from src import Tile
from src import image_utils


def enrich(tile, save_func, scale_min=4, scale_max=4):
    """
    Creates a list of new tiles obtained by cutting and gluing current tile.

    :param tile: starting tile
    :param save_func: function to write generated images; should be called as save_func(image, image_path)
    :param scale_min: how small minimum size of obtained tile can be relative to starting tile
    :param scale_max: how big maximum size of obtained tile can be relative to starting tile
    :return: None
    """

    min_size = tile.img.dims[0] // scale_min
    max_size = tile.img.dims[0] // scale_max

    def __save_tile(tile):
        for x in enrich_colour(tile):
            for y in enrich_contrast(x):
                img_name = str(uuid.uuid4())
                save_func(y.img, '{}.jpg'.format(img_name))

    to_fragment = deque()
    to_augment = deque()

    to_fragment.append(tile)

    while to_fragment:
        curr = to_fragment.popleft()

        __save_tile(curr)

        if curr.get_rhombus().dims[0] >= min_size:
            rhombus = curr.get_rhombus()
            to_fragment.append(rhombus)
            to_augment.append(rhombus)
            __save_tile(rhombus)

            no_center = curr.remove_center()
            to_fragment.append(no_center)
            to_augment.append(no_center)
            __save_tile(no_center)

        if curr.get_quadrant(0, 0).dims[0] >= min_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    __save_tile(curr.assemble_quadrant_unfold(i, j))

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(curr)

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
    img = tile.img
    channels = product([0, 1, 2], repeat=3)
    return [Tile.Tile(np.array([img[..., _[0]], img[..., _[1]], img[..., _[2]]]).transpose())
            for _ in channels
            if len(set(_)) > 1]


def enrich_contrast(tile):
    """
    Returns a list of tiles obtained from starting tile 
    by increasing contrast in different colour channels.

    :param tile: starting tile
    :return: list of re-contrasted tiles
    """
    img = tile.img
    channels = list(combinations([0, 1, 2], 1)) + list(combinations([0, 1, 2], 2))
    return [Tile.Tile(image_utils.increase_contrast(img, _)) for _ in channels]
