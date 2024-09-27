import pytest
from .utils import include_colibri
include_colibri()

from colibri.models import build_network, Autoencoder, Unet
import torch

model_list = ["unet","autoencoder"]

@pytest.fixture
def imsize():
    b = 1
    h = 32
    w = 32
    c = 1
    return b, c, h, w

def choose_model(name, imsize):

    if name == "autoencoder":
        model_layer = Autoencoder
    elif name == "unet":
        model_layer = Unet
    
    model = build_network(model=model_layer, in_channels=imsize[1], out_channels=imsize[1])
    return model

@pytest.mark.parametrize("model_name", model_list)
def test_model(model_name, imsize):
    
    model = choose_model(model_name, imsize)
    x = torch.randn(imsize)
    y = model(x)

    assert y.shape == x.shape