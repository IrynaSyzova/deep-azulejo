import uuid
from skimage.metrics import structural_similarity

from src import logging_utils, s3_utils, image_utils
from src.Tile import Tile

MIN_SIZE = 64
MAX_SIZE = 2500
MAX_SIMILARITY = 0.5

logger = logging_utils.get_logger(__name__)


def enrich(tile, key, max_fragmentation_depth=2, max_augmentation_depth=2, max_overall_depth=None):
    if max_overall_depth is None:
        max_overall_depth = max(4, max_fragmentation_depth+1, max_augmentation_depth+1)
    temp_key = '{}/temp'.format(key)
    __enrich(__save_tile(tile, temp_key), key, temp_key, max_fragmentation_depth, max_augmentation_depth,
             max_overall_depth)
    s3_utils.delete_from_s3(temp_key)


def __enrich(tile_path, key, temp_key, max_fragmentation_depth=2, max_augmentation_depth=2, max_overall_depth=4):
    logger.info('Fragmentation depth: {}, augmentation depth: {}, overall depth: {}'.format(
        max_fragmentation_depth, max_augmentation_depth, max_overall_depth
    ))

    imgs_in_key = len(s3_utils.get_image_list_from_s3(key))
    imgs_in_temp_key = len(s3_utils.get_image_list_from_s3(temp_key))

    logger.info('{} images created'.format(
        imgs_in_key - imgs_in_temp_key
    ))

    tile = __read_tile(tile_path)

    _ = __save_tile(tile, key)
    _ = __save_tile(tile.add_border_reflect(border_thickness=0.33), key)

    if max_overall_depth >= 0:

        if tile.dims[0] <= MIN_SIZE:
            max_fragmentation_depth = 0

        if max_fragmentation_depth > 0:

            fragments = [
                __save_tile(tile.get_rhombus(), temp_key),
                __save_tile(tile.get_quadrant(0, 0), temp_key),
            ]
            if __similarity(tile.get_quadrant(0,0), tile.get_quadrant(1, 0).rotate(clockwise=True)) <= MAX_SIMILARITY:
                fragments += [
                    __save_tile(tile.get_quadrant(0, 1).rotate(clockwise=False), temp_key),
                    __save_tile(tile.get_quadrant(1, 0).rotate(clockwise=True), temp_key),
                    __save_tile(tile.get_quadrant(1, 1).rotate().rotate(), temp_key)
                ]
            for fragment in fragments:
                __enrich(fragment, key, temp_key, max_fragmentation_depth - 1, max_augmentation_depth + 1,
                         max_overall_depth - 1)

        if max_augmentation_depth > 0:
            __enrich(__save_tile(tile.add_border_reflect(border_thickness=0.5), temp_key), key, temp_key,
                     max_fragmentation_depth, max_augmentation_depth - 1, max_overall_depth - 1)

            __enrich(__save_tile(tile.assemble_quadrant_unfold(0, 0), temp_key), key, temp_key,
                     max_fragmentation_depth + 1,
                     max_augmentation_depth - 1,
                     max_overall_depth - 1)

            if max_fragmentation_depth > 0 and max_overall_depth > 0:
                __enrich(__save_tile(tile.assemble_quadrant_unfold(0, 0).remove_center(), temp_key), key, temp_key,
                         max_fragmentation_depth - 1, max_augmentation_depth - 1, max_overall_depth - 1)


def __save_tile(tile_save, key):
    """
    Recolours and saves the tile at path key with randomly generated name
    :param tile_save: tile to save
    :param key: path to save image
    :return: path to saved file
    """
    if tile_save.dims[0] >= MAX_SIZE:
        tile_save = Tile(image_utils.resize(tile_save.img, (MAX_SIZE, MAX_SIZE)))
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


def __similarity(tile1, tile2):
    return structural_similarity(tile1.img, tile2.img, multichannel=True)
