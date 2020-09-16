import boto3
from PIL import Image
from io import BytesIO
import numpy as np

BUCKET = 'iryna-sandbox'

s3_resource = boto3.resource('s3')


def read_image_from_s3(key):
    """
    Reads image from s3
    :param key: path in s3 in bucket 'iryna-sandbox'
    :return: image as np.array
    """
    bucket = s3_resource.Bucket(BUCKET)
    object = bucket.Object(key)
    response = object.get()
    file_stream = response['Body']
    im = Image.open(file_stream)
    return np.array(im)


def write_image_to_s3(img_array, key):
    """
    Writes image to s3
    :param img_array: image as np.array
    :param key: path in s3 in bucket 'iryna-sandbox'
    :return: None
    """
    bucket = s3_resource.Bucket(BUCKET)
    object = bucket.Object(key)
    file_stream = BytesIO()
    im = Image.fromarray(img_array)
    im.save(file_stream, format='jpeg')
    object.put(Body=file_stream.getvalue())


def get_image_list_from_s3(key):
    """
    Returns list of images of .jpg format from prefix key
    :param key: path in s3 in bucket 'iryna-sandbox'
    :return: list of images located in the path
    """
    bucket = s3_resource.Bucket(BUCKET)
    bucket.objects.filter(Prefix=key)

    return [file.key for file in bucket.objects.filter(Prefix=key).all() if file.key.endswith('.jpg')]
