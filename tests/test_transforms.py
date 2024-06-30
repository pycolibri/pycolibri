import pytest
from torch.utils import data

from .utils import include_colibri
include_colibri()

import torch

from colibri.recovery.transforms import DCT2D

def load_img():
    from colibri.data.datasets import CustomDataset
    name = 'cifar10'
    path = '.'
    batch_size = 16

    builtin_dict = dict(train=True, download=True)
    dataset = CustomDataset(name, path,
                            builtin_dict=builtin_dict,
                            transform_dict=None)
    dataset_loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    sample = next(iter(dataset_loader))
    return sample


def test_dct2d():

    x_true = load_img()['input']
    transform_dct = DCT2D()

    theta = transform_dct.forward(x_true)
    x_hat = transform_dct.inverse(theta)

    mse = (x_true - x_hat).pow(2).mean().item()
    assert mse < 1e-6, f"Mean Squared Error: {mse}"

    
        