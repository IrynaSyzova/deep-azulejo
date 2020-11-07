import boto3
from PIL import Image
from io import BytesIO
import numpy as np

s3_resource = boto3.resource('s3')
BUCKET = s3_resource.Bucket('iryna-sandbox')


def read_image_from_s3(key, as_array=True):
    """
    Reads image from s3
    :param key: path in s3 in BUCKET
    :param as_array: whether to convert image to numpy array
    :return: image as np.array
    """
    object = BUCKET.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    if as_array:
        return np.array(im)
    return im


def write_image_to_s3(img_array, key):
    """
    Writes image to s3
    :param img_array: image as np.array
    :param key: path in s3 in BUCKET
    :return: None
    """
    object = BUCKET.Object(key)
    file_stream = BytesIO()
    im = Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())


def get_image_list_from_s3(key):
    """
    Returns list of images of .jpg format from prefix key
    :param key: path in s3 in BUCKET
    :return: list of images located in the path
    """

    BUCKET.objects.filter(Prefix=key)

    return [file.key for file in BUCKET.objects.filter(Prefix=key).all() if file.key.endswith('.jpg')]


def delete_from_s3(key):
    """
    Deletes every object that starts with key
    :param key: path in s3 in BUCKET
    :return: None
    """
    BUCKET.objects.filter(Prefix=key).delete()
