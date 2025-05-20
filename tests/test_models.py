import pytest
from .utils import include_colibri
include_colibri()

from colibri.models import build_network, Autoencoder, Unet
from colibri.models.learned_proximals import SparseProximalMapping
import torch.nn as nn
import torch
from colibri.recovery.terms.fidelity import L2
from colibri.models.unrolling import UnrollingFISTA
from colibri.recovery.terms.prior import LearnedPrior
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

def load_acqusition(img_size):
    from colibri.optics import SPC

    acquisition_config = dict(
        input_shape=img_size,
    )

    n_measurements = 25 ** 2
    acquisition_config['n_measurements'] = n_measurements

    acquisiton_model = SPC(**acquisition_config)
    return acquisiton_model

def load_img():
    from colibri.data.datasets import CustomDataset
    name = 'fashion_mnist'
    path = '.'


    dataset = CustomDataset(name, path)
    sample = dataset[0]['input']
    return sample.unsqueeze(0)

@pytest.mark.parametrize("model_name", model_list)
def test_model(model_name, imsize):
    
    model = choose_model(model_name, imsize)
    x = torch.randn(imsize)
    y = model(x)

    assert y.shape == x.shape

def test_unrolling_model():
    x_true = load_img()
    img_size = x_true.shape[1:]
    acquisition_model = load_acqusition(img_size)


    fidelity = L2()
    prior_args ={'autoencoder_args': {'in_channels': 1, 'out_channels': 1, 'feautures': [1,1,1,1]},'beta': 1e-3}
    algo_params = {
        'max_iters': 10,
        'alpha': 1e-4,
        '_lambda': 0.001,
    }
    models = nn.Sequential(*[SparseProximalMapping(**prior_args) for _ in range(algo_params["max_iters"])])

    y = acquisition_model(x_true)
    x_trivial = acquisition_model(y, type_calculation="backward")
    fista_unrolling = UnrollingFISTA(acquisition_model, fidelity, **algo_params, models=models)

    x_hat = fista_unrolling(y, x0=x_trivial)

    # Check if the output has the same shape as the input
    assert x_true.shape == x_hat.shape, f"Shape of the input: {x_true.shape}, Shape of the output: {x_hat.shape}"

    error_trivial = torch.norm(x_true - x_trivial)
    error_algo = torch.norm(x_true - x_hat)

    # Check if the error of the algorithm is smaller than the error of the trivial solution
    assert error_algo < error_trivial, f"Error of the algorithm: {error_algo}, Error of the trivial solution: {error_trivial}"
