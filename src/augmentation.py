import numpy as np

from collections import deque
from itertools import product

from src import Tile


def rearrange(tile, min_size, max_size):
    """
    Creates a list of new tiles obtained by cutting and gluing current tile.

    :param tile: starting tile
    :param min_size: minimum size of obtained tile
    :param max_size: maximum size of obtained tile
    :return: list of generated tiles
    """
    to_fragment = deque()
    to_augment = deque()
    fragments = deque()

    to_fragment.append(tile)

    while to_fragment:
        curr = to_fragment.popleft()

        fragments.append(curr)

        if curr.get_rhombus().dims[0] >= min_size:
            fragments.append(curr.get_rhombus())
            to_fragment.append(curr.get_rhombus())
            to_augment.append(curr.get_rhombus())

        if curr.get_quadrant(0, 0).dims[0] >= min_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    fragments.append(curr.assemble_quadrant_unfold(i, j))

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(curr)

    while to_augment:
        curr = to_augment.pop()
        fragments.append(curr.get_rhombus())

        if curr.dims[0] * 4 <= max_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    fragments.append(curr.assemble_quadrant_unfold(i, j))
                    to_augment.append(curr.assemble_quadrant_unfold(i, j))

            fragments.append(curr.assemble_quadrant_windmill())
            to_augment.append(curr.assemble_quadrant_windmill())

    return fragments


def colour(tile):
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


def enrich(tile, min_size, max_size):
    """
    Rearranges and recolours starting tile to obtained a list of new tiles.

    :param tile: starting tile
    :param min_size: minimum size of obtained tile
    :param max_size: maximum size of obtained tile
    :return: list of generated tiles
    """
    rearranged = deque(rearrange(tile, min_size, max_size))
    result = deque()
    while rearranged:
        curr = rearranged.pop()
        result.extend(colour(curr))

    return list(result)
