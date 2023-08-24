import tensorflow as tf
import numpy as np

def prism_operator(x, shift_sign = 1):
    """
    Prism operator, it shifts the input tensor x according to spectral shift made by a prism
    :param x: Input tensor with shape (1, M, N, L)
    :param shift_sign: Integer, it can be 1 or -1, it indicates the direction of the shift
    :return: Output tensor with shape (1, M, N + L - 1, L) if shift_sign is 1, or (1, M, N-L+1, L) if shift_sign is -1
    """

    assert shift_sign == 1 or shift_sign == -1, "The shift sign must be 1 or -1"
    _, M, N, L = x.shape  # Extract spectral image shape

    x = tf.unstack(x, axis=-1)


    if shift_sign == 1:
        # Shifting produced by the prism 
        x = [tf.pad(x[l], [(0, 0), (0, 0), (l, L - l - 1)]) for l in range(L)]
    else:
        # Unshifting produced by the prism
        x = [x[l][:, :, l:N - (L- 1)+l] for l in range(L)]

    x = tf.stack(x, axis=-1)
    return x

def forward_color_cassi(x, ca):
    """
    Forward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    :param x: Spectral image with shape (1, M, N, L)
    :param ca: Coded aperture with shape (1, M, N, L)
    :return: Measurement with shape (1, M, N, 1)
    """
    y = tf.multiply(x, ca)
    return tf.reduce_sum(y, axis=-1, keepdims=True)

def backward_color_cassi(y, ca):
    """
    Backward operator of color coded aperture snapshot spectral imager (Color-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    :param y: Measurement with shape (1, M, N, L)
    :param ca: Coded aperture with shape (1, M, N, L)
    :return: Spectral image with shape (1, M, N, L)
    """
    x = tf.multiply(y, ca)
    return x


def forward_dd_cassi(x, ca):
    """
    Forward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    :param x: Spectral image with shape (1, M, N, L)
    :param ca: Coded aperture with shape (1, M, N + L - 1, 1)
    :return: Measurement with shape (1, M, N + L - 1, 1)
    """
    _, M, N, L = x.shape  # Extract spectral image shape
    assert ca.shape[-2] == N + L - 1, "The coded aperture must have the same size as a dispersed scene"
    ca = tf.tile(ca, [1, 1, 1, L])
    ca = prism_operator(ca, shift_sign = -1)
    y = forward_color_cassi(x, ca)
    return y


def backward_dd_cassi(y, ca):
    """
    Backward operator of dual disperser coded aperture snapshot spectral imager (DD-CASSI), more information refer to: Computational snapshot multispectral cameras: Toward dynamic capture of the spectral world https://doi.org/10.1109/MSP.2016.2582378
    :param y: Measurement with shape (1, M, N + L - 1, 1)
    :param ca: Coded aperture with shape (1, M, N + L - 1, 1)
    :return: Spectral image with shape (1, M, N, L)
    """
    _, M, N, _ = ca.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = tf.tile(y, [1, 1, 1, L])
    ca = tf.tile(ca, [1, 1, 1, L])
    ca = prism_operator(ca, shift_sign = -1)
    return backward_color_cassi(y, ca)

def forward_cassi(x, ca):
    """
    Forward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    :param x: Spectral image with shape (1, M, N, L)
    :param ca: Coded aperture with shape (1, M, N, 1)
    :return: Measurement with shape (1, M, N + L - 1, 1)
    """
    y1 = tf.multiply(x, ca)  # Multiplication of the scene by the coded aperture
    _, M, N, L = y1.shape  # Extract spectral image shape
    # shift and sum
    y2 = prism_operator(y1, shift_sign = 1)
    y2 = tf.reduce_sum(y2, axis=-1, keepdims=True)
    return y2


def backward_cassi(y, ca):
    """
    Backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    :param y: Measurement with shape (1, M, N + L - 1, 1)
    :param ca: Coded aperture with shape (1, M, N, 1)
    :return: Spectral image with shape (1, M, N, L)
    """
    _, M, N, _ = y.shape  # Extract spectral image shape
    L = N - M + 1  # Number of shifts
    y = tf.tile(y, [1, 1, 1, L])
    y = prism_operator(y, shift_sign = -1)
    return tf.multiply(y, ca)
