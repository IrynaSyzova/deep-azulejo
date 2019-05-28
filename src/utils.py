import numpy as np

from collections import deque
from itertools import product

from src import Tile


def rearrange(tile, min_size, max_size):
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
            fragments.append(curr.get_quadrant(0, 0))
            fragments.append(curr.get_quadrant(0, 1))
            fragments.append(curr.get_quadrant(1, 0))
            fragments.append(curr.get_quadrant(1, 1))

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(curr)

    while to_augment:
        curr = to_augment.pop()
        if curr.dims[0] * 4 <= max_size:
            fragments.append(curr.assemble_quadrant_radial(0, 0))
            fragments.append(curr.assemble_quadrant_radial(1, 0))
            fragments.append(curr.assemble_quadrant_radial(0, 1))
            fragments.append(curr.assemble_quadrant_radial(1, 1))
            fragments.append(curr.assemble_quadrant_circular())

    return fragments


def colour(tile):
    img = tile.img
    channels = product([0, 1, 2], repeat=3)
    return [Tile.Tile(np.array([img[..., _[0]], img[..., _[1]], img[..., _[2]]]).transpose())
            for _ in channels
            if len(set(_)) > 1]


def enrich(tile, min_size, max_size):
    rearranged = deque(rearrange(tile, min_size, max_size))
    result = deque()
    while rearranged:
        curr = rearranged.pop()
        result.extend(colour(curr))

    return list(result)