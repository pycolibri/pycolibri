import pytest
from .test_utils import include_colibri
include_colibri()

from colibri_hdsp.models import build_network, Autoencoder, Unet
import tensorflow as tf

model_list = ["autoencoder", "unet"]

@pytest.fixture
def imsize():
    b = 1
    h = 32
    w = 32
    c = 1
    return b, h, w, c

def choose_model(name, imsize):

    if name == "autoencoder":
        model_layer = Autoencoder
    elif name == "unet":
        model_layer = Unet
    
    model = build_network(model=model_layer, size=imsize[-2], in_channels=imsize[-1], out_channels=imsize[-1])
    return model

@pytest.mark.parametrize("model_name", model_list)
def test_model(model_name, imsize):
    
    model = choose_model(model_name, imsize)
    x = tf.random.normal(imsize)
    y = model(x)

    assert y.shape == x.shape

