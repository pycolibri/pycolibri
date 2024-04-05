import pytest
from .utils import include_colibri
include_colibri()


import torch
from colibri_hdsp.optics.cassi import SD_CASSI, DD_CASSI, C_CASSI
from colibri_hdsp.optics.spc import SPC

@pytest.fixture
def imsize():
    b = 8
    h = 32
    w = 32
    c = 3
    return b, c, h, w

def cassi_config(imsize, mode):
    b, c, h, w = imsize#[1:]

    if mode == "sd_cassi":
        out = (b, 1, h, w + c - 1)
    elif mode == "dd":
        out = (b, 1, h, w)
    elif mode == "color":
        out = (b, 1, h, w)
    
    return out

mode_list = ["sd_cassi", "dd", "color"]

@pytest.mark.parametrize("mode", mode_list)
def test_cassi(mode, imsize):
    cube = torch.randn(imsize)
    out_shape = cassi_config(imsize, mode)

    if mode == "sd_cassi":
        cassi = SD_CASSI(imsize[1:])
    elif mode == "dd":
        cassi = DD_CASSI(imsize[1:])
    elif mode == "color":
        cassi = C_CASSI(imsize[1:])

    cube = cube.float()
    measurement = cassi(cube, type_calculation="forward")
    backward = cassi(measurement, type_calculation="backward")
    forward_backward = cassi(cube, type_calculation="forward_backward")
    assert measurement.shape == out_shape
    assert backward.shape == cube.shape
    assert forward_backward.shape == cube.shape


@pytest.fixture
def spc_config():
    img_size = [128, 32, 32]
    n_measurements = 256
    return img_size, n_measurements

def test_spc_forward(spc_config):
    img_size, n_measurements = spc_config
    spc = SPC(img_size, n_measurements)

    b, c, h, w = 1, *img_size
    x = torch.randn(b, c, h, w)

    y_forward = spc(x, type_calculation="forward")
    expected_shape = (b, n_measurements, c) 
    assert y_forward.shape == expected_shape, "Forward output shape is incorrect"