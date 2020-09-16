import boto3
import cv2

BUCKET = 'iryna-sandbox'

s3_resource = boto3.client('s3')


def save_img(img, filepath, filename):
    _, buffer = cv2.imencode('.jpg', img[...,::-1])
    path = '{}/{}'.format(filepath, filename)
    s3_resource.put_object(Body=buffer.tostring(), Bucket=BUCKET, Key=path)
    