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
        logger.debug('Finished saving.')
        return img_path

    def __read_tile(key):
        """
        Reads image from s3 and converts it to Tile type
        :param key: path to read image from
        :return: Tile
        """
        return Tile(s3_utils.read_image_from_s3(key))

    to_fragment, to_fragment_key = deque(), '{}/to_fragment'.format(key)
    to_augment, to_augment_key = deque(), '{}/to_augment'.format(key)

    to_fragment.append(__save_tile(tile, to_fragment_key))

    while to_fragment:
        curr = __read_tile(to_fragment.popleft())

        _ = __save_tile(curr, key)

        if curr.dims[0] / 2 * 2**2 >= min_size:
            rhombus = curr.get_rhombus()
            if rhombus.dims[0] >= min_size:
                to_fragment.append(__save_tile(rhombus, to_fragment_key))
                to_augment.append(__save_tile(rhombus, to_augment_key))

            _ = __save_tile(rhombus, key)

        if curr.dims[0] / 2 >= min_size:
            quadrants = [
                curr.get_quadrant(0, 0),
                curr.get_quadrant(0, 1).rotate(clockwise=False),
                curr.get_quadrant(1, 0).rotate(clockwise=True),
                curr.get_quadrant(1, 1).rotate().rotate(),
            ]
            for quadrant in quadrants:
                to_fragment.append(__save_tile(quadrant, to_fragment_key))
                to_augment.append(__save_tile(quadrant, to_augment_key))
                _ = __save_tile(quadrant, key)

        if curr.dims[0] * 4 <= max_size:
            to_augment.append(__save_tile(curr, to_augment_key))

        logger.info('{} files in fragmentation queue, {} files in augmentation queue'.format(
            len(to_fragment), len(to_augment)
        ))

    while to_augment:
        curr = __read_tile(to_augment.pop())
        while curr.dims[0] >= max_size:
            curr = __read_tile(to_augment.pop())

        reflect_border_thickness = 0.5
        if curr.dims[0] * (1+2*reflect_border_thickness) <= max_size:
            border_reflect = curr.add_border_reflect(border_thickness=reflect_border_thickness)
            border_reflect_rhombus = border_reflect.get_rhombus()

            _ = __save_tile(border_reflect, key)
            _ = __save_tile(border_reflect_rhombus, key)

            to_augment.append(__save_tile(border_reflect_rhombus, to_augment_key))

        reflect_border_thickness = 0.33
        if curr.dims[0] * (1 + 2 * reflect_border_thickness) <= max_size:
            border_reflect = curr.add_border_reflect(border_thickness=reflect_border_thickness)
            # to_augment.append(__save_tile(border_reflect, to_augment_key))

            _ = __save_tile(border_reflect, key)
            # _ = __save_tile(border_reflect.remove_center(), key)

        reflect_border_thickness = 0.25
        if curr.dims[0] * (1 + 2 * reflect_border_thickness) <= max_size:
            border_reflect = curr.add_border_reflect(border_thickness=reflect_border_thickness)
            # to_augment.append(__save_tile(border_reflect, to_augment_key))
            _ = __save_tile(border_reflect, key)

            if border_reflect.dims[0] * 2 <= max_size:
                border_reflect_quadrant = border_reflect.assemble_quadrant_unfold(0, 0)
                _ = __save_tile(border_reflect_quadrant, key)
                to_augment.append(__save_tile(border_reflect_quadrant, to_augment_key))

        if curr.dims[0] * 2 <= max_size:
            unfolded = curr.assemble_quadrant_unfold(0, 0)
            # to_augment.append(__save_tile(unfolded, to_augment_key))

            _ = __save_tile(unfolded, key)
            _ = __save_tile(unfolded.get_rhombus(), key)
            _ = __save_tile(unfolded.remove_center(), key)
            to_augment.append(__save_tile(unfolded, to_augment_key))

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
