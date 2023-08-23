import tensorflow as tf


def normalize_img(image, max_val=255.):
    """
    Functions that normalizes images.
    :param image: image to normalize
    :param max_val: max value of the image
    :return: normalized image
    """
    norm_img = tf.cast(image, tf.float32) / max_val
    return norm_img


def hot_encode(label, num_classes=10):
    """
    Function that Hot encode labels
    :param label: label to encode
    :param num_classes: number of classes
    :return: encoded label
    """
    return tf.one_hot(label, num_classes)
