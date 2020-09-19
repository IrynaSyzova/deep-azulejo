import uuid

from src import logging_utils, s3_utils
from src.Tile import Tile

logger = logging_utils.get_logger(__name__)


def enrich_wrapper(tile, key, max_fragmentation_depth=2, max_augmentation_depth=2):
    temp_key = '{}/temp'.format(key)
    enrich(__save_tile(tile, temp_key), key, max_fragmentation_depth, max_augmentation_depth)
    s3_utils.delete_from_s3(temp_key)


def enrich(tile_path, key, max_fragmentation_depth=2, max_augmentation_depth=2):
    temp_key = '{}/temp'.format(key)

    tile = __read_tile(tile_path)

    if max_fragmentation_depth >= 1:
        fragments = [
            __save_tile(tile.get_rhombus(), temp_key),
            __save_tile(tile.get_quadrant(0, 0), temp_key),
            __save_tile(tile.get_quadrant(0, 1).rotate(clockwise=False), temp_key),
            __save_tile(tile.get_quadrant(1, 0).rotate(clockwise=True), temp_key),
            __save_tile(tile.get_quadrant(1, 1).rotate().rotate(), temp_key),
        ]
        for fragment in fragments:
            enrich(fragment, key, max_fragmentation_depth - 1, max_augmentation_depth)

    if max_augmentation_depth >= 1:
        augments = [
            __save_tile(tile.add_border_reflect(border_thickness=0.5), temp_key),
            __save_tile(tile.assemble_quadrant_unfold(0, 0), temp_key),
            __save_tile(tile.assemble_quadrant_unfold(0, 0).remove_center(), temp_key)
        ]
        for fragment in augments:
            enrich(fragment, key, max_fragmentation_depth, max_augmentation_depth - 1)

    _ = __save_tile(tile, key)
    _ = __save_tile(tile.add_border_reflect(border_thickness=0.25), key)
    _ = __save_tile(tile.add_border_reflect(border_thickness=0.33), key)

    return


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
