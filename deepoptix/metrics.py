import tensorflow as tf

"""
metrics.py
====================================
The metrics
"""

def psnr(y_true, y_pred):
    """Calculate Peak Signal to Noise Ratio between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        PSNR between y_true and y_pred.
    """
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def ssim(y_true, y_pred):
    """Calculate Structural Similarity Index between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        SSIM between y_true and y_pred.
    """
    return tf.image.ssim(y_true, y_pred, max_val=1.0)

def mse(y_true, y_pred):
    """Calculate Mean Squared Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MSE between y_true and y_pred.
    """
    return tf.keras.losses.mean_squared_error(y_true, y_pred) 

def mae(y_true, y_pred):
    """Calculate Mean Absolute Error between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        MAE between y_true and y_pred.
    """
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def accuracy(y_true, y_pred, num_classes=2):
    """Calculate accuracy between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.
        num_classes: Number of classes.

    Returns:
        Accuracy between y_true and y_pred.
    """
    if num_classes == 2:
        return tf.keras.metrics.binary_accuracy(y_true, y_pred)
    else:
        return tf.keras.metrics.categorical_accuracy(y_true, y_pred)
  
def precision(y_true, y_pred):
    """Calculate precision between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        Precision between y_true and y_pred.
    """
    return tf.keras.metrics.precision(y_true, y_pred)

def recall(y_true, y_pred):
    """Calculate recall between y_true and y_pred.

    Args:
        y_true: Tensor of actual values.
        y_pred: Tensor of predicted values.

    Returns:
        Recall between y_true and y_pred.
    """
    return tf.keras.metrics.recall(y_true, y_pred)

def SAM(y_true, y_pred, reduce=None):
    """Calculate Spectral Angle Mapper between org and pred.

    Args:
        y_true: Tensor of actual values. [B x H x W x C] 
        y_pred: Tensor of predicted values. [B x H x W x C]

    Returns:
        SAM between org and pred.
    """
    numerator = tf.sum(tf.math.multiply(y_pred, y_true), axis=-1)
    denominator = tf.norm(y_true, ord=2, axis=-1) * tf.norm(y_pred, ord=2, axis=-1)
    val = tf.clip_by_value(numerator / denominator, -1, 1)
    angles = tf.math.acos(val)

    if reduce is None:
        return angles
    elif reduce == 'mean':
        return tf.reduce_mean(angles)
    elif reduce == 'sum':
        return tf.reduce_sum(angles)
    