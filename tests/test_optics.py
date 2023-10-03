import pytest
from .utils import include_colibri
include_colibri()


import tensorflow as tf
from colibri_hdsp.optics.cassi import CASSI

@pytest.fixture
def imsize():
    b = 1
    h = 32
    w = 32
    c = 31
    return b, h, w, c

def compute_outshape(imsize, mode):
    h, w, c = imsize[1:]

    if mode == "base":
        out = (1, h, w + c - 1, 1)
    elif mode == "dd":
        out = (1, h, w, 1)
    
    return out

mode_list = ["base", "dd"]

@pytest.mark.parametrize("mode", mode_list)
def test_cassi(mode, imsize):

    cube      =  tf.random.normal(imsize)
    out_shape = compute_outshape(imsize, mode)

    h, w, c = imsize[1:]

    cassi = CASSI(mode=mode)
    cassi.build((h, w, c))

    cube_tf = tf.convert_to_tensor(cube, dtype=tf.float32)
    measurement = cassi(cube_tf, type_calculation="forward")

    assert measurement.shape == out_shape

