import numpy as np
import uuid
import cv2

from collections import deque
from itertools import product, combinations

from src import Tile, logging_utils

logger = logging_utils.get_logger(__name__)

SOLID_BORDER_THICKNESS = 0.05
REFLECT_BORDER_THICKNESS = 0.5


def enrich(tile, save_func, scale_min=0.25, scale_max=4, max_imgs=5000):
    """
    Creates a list of new tiles obtained by cutting and gluing current tile.

    :param tile: starting tile
    :param save_func: function to write generated images; should be called as save_func(image, image_path)
    :param scale_min: how small minimum size of obtained tile can be relative to starting tile by dimension
    :param scale_max: how big maximum size of obtained tile can be relative to starting tile by dimension
    :param max_imgs: when reached max_imgs, function will exit
    :return: None
    """

    min_size = int(tile.dims[0] * scale_min)
    max_size = int(tile.dims[0] * scale_max)

    logger.info('Enriching tile of {x}x{x} dims; generated tiles will be from {y}x{y} to {z}x{z}'.format(
        x=tile.dims[0],
        y=min_size,
        z=max_size
    ))

    counter = 0

    def __save_tile(tile_save):
        """
        Recolours and saves the tile, returns number of tiles saved
        :param tile_save: tile to save
        :return: number of tiles saved
        """
        tile_colours = enrich_colour(tile_save)
        for x in tile_colours:
            img_name = str(uuid.uuid4())
            save_func(x.img, '{}.jpg'.format(img_name))
        return len(tile_colours)

    to_fragment = deque()
    to_augment = deque()

    to_fragment.append(tile)

    while to_fragment:
        curr = to_fragment.popleft()

        counter += __save_tile(curr)
        if counter >= max_imgs:
            return

        if curr.dims[0] / 2 * 2**2 >= min_size:
            rhombus = curr.get_rhombus()
            if rhombus.dims[0] >= min_size:
                to_fragment.append(rhombus)
                to_augment.append(rhombus)

                counter += __save_tile(rhombus)
                if counter >= max_imgs:
                    return

        if curr.dims[0] / 3 * 2 >= min_size:
            no_center = curr.remove_center()
            if no_center.dims[0] >= min_size:
                to_fragment.append(no_center)
                to_augment.append(no_center)

            counter += __save_tile(no_center)
            if counter >= max_imgs:
                return

        if curr.dims[0] / 2 >= min_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    quadrant = curr.get_quadrant(i, j)
                    to_fragment.append(quadrant)
                    to_augment.append(quadrant)
                    counter += __save_tile(quadrant)
                    if counter >= max_imgs:
                        return

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(curr)

        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))

    while to_augment:
        curr = to_augment.pop()

        if curr.dims[0] * (1+2*REFLECT_BORDER_THICKNESS) <= max_size:
            border_reflect = curr.add_border(border_thickness=REFLECT_BORDER_THICKNESS, border_type=cv2.BORDER_REFLECT)
            to_augment.append(border_reflect)

            counter += __save_tile(border_reflect)
            if counter >= max_imgs:
                return

        if curr.dims[0] * 4 <= max_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    unfolded = curr.assemble_quadrant_unfold(i, j)
                    to_augment.append(unfolded)

                    counter += __save_tile(unfolded)
                    if counter >= max_imgs:
                        return

                    unfolded_rhombus = unfolded.get_rhombus()
                    # to_augment.append(unfolded_rhombus)

                    counter += __save_tile(unfolded_rhombus)
                    if counter >= max_imgs:
                        return

                    unfolded_no_center = unfolded.remove_center()
                    # to_augment.append(unfolded_no_center)

                    counter += __save_tile(unfolded_no_center)
                    if counter >= max_imgs:
                        return


            windmill = curr.assemble_quadrant_windmill()
            to_augment.append(windmill)

            counter += __save_tile(windmill)
            if counter >= max_imgs:
                return

            windmill_rhombus = windmill.get_rhombus()
            # to_augment.append(windmill_rhombus)

            counter += __save_tile(windmill_rhombus)
            if counter >= max_imgs:
                return

            windmill_no_center = windmill.remove_center()
            to_augment.append(windmill_no_center)

            # counter += __save_tile(windmill_no_center)
            if counter >= max_imgs:
                return
        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))


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
