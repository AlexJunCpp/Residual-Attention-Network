import math
import numpy as np
import tensorflow as tf


crop_size = (32, 32)


def image_crop(img):
    """
    Randomly crops the image.
    Inputs:
        img: img to crop.
    Outputs:
        img: img after crop.
    """
    # Random pad and crop
    crop_h = crop_size[0]
    crop_w = crop_size[1]
    img = tf.image.pad_to_bounding_box(img, 4, 4, 40, 40)
    img = tf.random_crop(img, [crop_h, crop_w, 3])
    return img


def image_mirror(img):
    """
    Randomly mirrors the image.
    Inputs:
        img: img to mirror.
    Outputs:
        img: img after mirror.
    """
    distort_left_right_random = tf.random_uniform([], 0, 1.0, dtype=tf.float32)
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    return img


def read_data_list(data_list):
    """
    Function to read data list
    """
    f = open(data_list, 'r')
    imgs = []
    labs = []
    for line in f:
        try:
            img = line.strip("\n").split(' ')[0]
            lab = line.strip("\n").split(' ')[1]
        except ValueError:
            img = lab = line.strip("\n")
        imgs.append(img)
        labs.append(lab)
    return imgs, labs


def read_images_from_disk(input_queue, is_training):
    """
    Read image, and label with optional pre-processing.
    """
    # Read files
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(input_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={'img_raw' : tf.FixedLenFeature([], tf.string),
                                                 'label': tf.FixedLenFeature([], tf.int64)
                                       })
    # Decode files
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    lab = tf.cast(features['label'], tf.int32)
    # RGB -> BGR
    img = tf.reshape(img, [32, 32, 3])
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    if is_training:
        # Whiten.
        img -= 127
        # Random mirror
        img = image_mirror(img)
        # Random crop
        img = image_crop(img)
    # Set static shape so that tensorflow knows shape at compile time.
    img.set_shape([32, 32, 3])
    lab.set_shape([])
    return img, lab


class Reader(object):
    """
    Generic Data Reader which reads images, normals and masks
    from the disk, and enqueues them into a TensorFlow queue.
    """

    def __init__(self, coord, data_list, is_training=True):
        """
        Initialize a Reader.
        """
        self.coord = coord
        self.data_list = data_list
        self.queue = tf.train.string_input_producer([data_list])
        self.img, self.lab = read_images_from_disk(self.queue, is_training)

    def dequeue(self, num_elements):
        img_bat, lab_bat = tf.train.batch([self.img, self.lab], num_elements)
        return img_bat, lab_bat
