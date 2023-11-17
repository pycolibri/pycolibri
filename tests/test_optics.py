import pytest
from .utils import include_colibri
include_colibri()


import torch
from colibri_hdsp.optics.cassi import CASSI
from colibri_hdsp.optics.spc import SPC

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
    elif mode == "color":
        out = (1, h, w, 1)
    
    return out

mode_list = ["base", "dd", "color"]

@pytest.mark.parametrize("mode", mode_list)
def test_cassi(mode, imsize):

    cube = torch.randn(imsize)
    out_shape = compute_outshape(imsize, mode)


    cassi = CASSI(imsize[1:], mode, "cpu")

    cube = cube.float()
    measurement = cassi(cube, type_calculation="forward")

    assert measurement.shape == out_shape


@pytest.fixture
def spc_config():
    img_size = 32
    m = 256
    return img_size, m

def test_spc_forward(spc_config):
    img_size, m = spc_config
    spc = SPC(img_size, m)

    b, c, h, w = 1, 3, img_size, img_size 
    x = torch.randn(b, c, h, w)

    y_forward = spc(x, type_calculation="forward")
    expected_shape = (b, m, c) 
    assert y_forward.shape == expected_shape, "Forward output shape is incorrect"