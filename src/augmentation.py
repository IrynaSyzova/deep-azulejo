import uuid
import cv2

from collections import deque
from itertools import product

from src import logging_utils, s3_utils
from src.Tile import Tile

logger = logging_utils.get_logger(__name__)


def enrich(tile, key, scale_min=0.25, scale_max=4):
    """
    Creates a list of new tiles obtained by cutting and gluing current tile;
    saves resulting images to s3

    :param tile: starting tile
    :param key: path in s3 to save generated images
    :param scale_min: how small minimum size of obtained tile can be relative to starting tile by dimension
    :param scale_max: how big maximum size of obtained tile can be relative to starting tile by dimension
    :return: None
    """

    min_size = int(tile.dims[0] * scale_min)
    max_size = int(tile.dims[0] * scale_max)

    logger.info('Enriching tile of {x}x{x} dims; generated tiles will be from {y}x{y} to {z}x{z}'.format(
        x=tile.dims[0],
        y=min_size,
        z=max_size
    ))

    def __save_tile(tile_save, key):
        """
        Recolours and saves the tile at path key with randomly generated name
        :param tile_save: tile to save
        :param key: path to save image
        :return: path to saved file
        """
        logger.debug('Saving img of {} dims.'.format(tile_save.dims))
        img_path = '{}/{}.jpg'.format(key, str(uuid.uuid4()))
        s3_utils.write_image_to_s3(tile_save.img, img_path)
        logger.info('Finished saving.')
        return img_path

    to_fragment, to_fragment_key = deque(), '{}/to_fragment'.format(key)
    to_augment, to_augment_key = deque(), '{}/to_augment'.format(key)

    to_fragment.append(__save_tile(tile, to_fragment_key))

    while to_fragment:
        curr = s3_utils.read_image_from_s3(to_fragment.popleft())

        _ = __save_tile(curr, key)

        if curr.dims[0] / 2 * 2**2 >= min_size:
            rhombus = curr.get_rhombus()
            if rhombus.dims[0] >= min_size:
                to_fragment.append(__save_tile(rhombus, to_fragment_key))
                to_augment.append(__save_tile(rhombus, to_augment_key))

            _ = __save_tile(rhombus, key)

        if curr.dims[0] / 3 * 2 >= min_size:
            no_center = curr.remove_center()
            if no_center.dims[0] >= min_size:
                to_fragment.append(__save_tile(no_center, to_fragment_key))
                to_augment.append(__save_tile(no_center, to_augment_key))
            _ = __save_tile(no_center, key)

        if curr.dims[0] / 2 >= min_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    quadrant = curr.get_quadrant(i, j)
                    to_fragment.append(__save_tile(quadrant, to_fragment_key))
                    to_augment.append(__save_tile(quadrant, to_augment_key))
                    _ = __save_tile(quadrant, key)

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(__save_tile(curr, to_augment_key))

        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))

    while to_augment:
        curr = s3_utils.read_image_from_s3(to_augment.pop())

        reflect_border_thickness = 0.5
        if curr.dims[0] * (1+2*reflect_border_thickness) <= max_size:
            border_reflect = curr.add_border(border_thickness=reflect_border_thickness, border_type=cv2.BORDER_REFLECT)
            # to_augment.append(__save_tile(border_reflect, to_augment_key))

            _ = __save_tile(border_reflect, key)
            _ = __save_tile(border_reflect.get_rhombus(), key)

        reflect_border_thickness = 0.33
        if curr.dims[0] * (1 + 2 * reflect_border_thickness) <= max_size:
            border_reflect = curr.add_border(border_thickness=reflect_border_thickness, border_type=cv2.BORDER_REFLECT)
            # to_augment.append(__save_tile(border_reflect, to_augment_key))

            _ = __save_tile(border_reflect, key)
            _ = __save_tile(border_reflect.remove_center(), key)

        if curr.dims[0] * 4 <= max_size:
            for i in range(0, 2):
                for j in range(0, 2):
                    unfolded = curr.assemble_quadrant_unfold(i, j)
                    # to_augment.append(__save_tile(unfolded, to_augment_key))

                    _ = __save_tile(unfolded, key)
                    _ = __save_tile(unfolded.get_rhombus(), key)
                    _ = __save_tile(unfolded.remove_center(), key)

        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))

    s3_utils.delete_from_s3(to_augment_key)
    s3_utils.delete_from_s3(to_fragment_key)


def enrich_colour(tile):
    """
    Returns a list of tiles obtained from starting tile by recolouring it.

    :param tile: starting tile
    :return: list of recoloured tiles
    """
    channels = product([0, 1, 2], repeat=3)
    return [recolour_tile(tile, _)
            for _ in channels]


def recolour_tile(tile, channels):
    """
    Recolours tile by using colour channels[0] for r, channels[1] for g, and channels[2] for b colour channels
    :param tile: tile to recolour
    :param channels: colour channels
    :return: recoloured tile
    """
    img = tile.img.copy()
    for i in (0, 1, 2):
        img[..., i] = tile.img[..., channels[i]]
    return Tile(img)
