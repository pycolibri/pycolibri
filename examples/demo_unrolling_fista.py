r"""
Demo FISTA.
===================================================

"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os

from torch.utils import data

os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory ", os.getcwd())

import sys

sys.path.append(os.path.join(os.getcwd()))

# General imports
import matplotlib.pyplot as plt
import torch
import os

# Set random seed for reproducibility
torch.manual_seed(0)

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())

if manual_device:
    device = manual_device
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Load dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset

name = "cifar10"
path = "."
batch_size = 1

builtin_dict = dict(train=True, download=True)
dataset = CustomDataset(name, path, builtin_dict=builtin_dict, transform_dict=None)

acquisition_name = "spc"  # ['spc', 'sd_cassi', 'dd_cassi', 'c_cassi']

# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = dataset[0]["input"]
sample = sample.unsqueeze(0).to(device)

# %%
# Optics forward model

import math
from colibri.optics import SPC, SD_CASSI, DD_CASSI, C_CASSI


img_size = sample.shape[1:]

acquisition_config = dict(
    input_shape=img_size,
)

if acquisition_name == "spc":
    n_measurements = 25**2
    n_measurements_sqrt = int(math.sqrt(n_measurements))
    acquisition_config["n_measurements"] = n_measurements

acquisition_model = {"spc": SPC, "sd_cassi": SD_CASSI, "dd_cassi": DD_CASSI, "c_cassi": C_CASSI}[
    acquisition_name
]

acquisition_model = acquisition_model(**acquisition_config)

y = acquisition_model(sample)

# Reconstruct image
import colibri
from colibri import models
print('dir(models) ->', dir(models))  
from colibri import recovery


print('dir(recover) ->', dir(recovery.terms.prior))  
from colibri.recovery.fista import Fista
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.models.learned_proximals import LearnedPrior
from colibri.models.unrolling import UnrollingFISTA
from colibri.models.autoencoder import Autoencoder



algo_params = {
    "max_iters": 10,
    "alpha": 1e-4,
    "_lambda": 0.01,
}

fidelity = L2()
prior = Sparsity(basis="dct")
prior_args ={'in_channels': 1, 'out_channels': 1, 'feautures': [32, 64, 128, 256]}
model = Autoencoder
prior = LearnedPrior(max_iter=algo_params["max_iters"], model=model, prior_args=prior_args)

fista_unrolling = UnrollingFISTA(acquisition_model, fidelity, **algo_params, model=model, prior_args=prior_args)

x0 = acquisition_model.forward(y, type_calculation="backward")

x_hat = fista_unrolling(y, x0=x0)


print(x_hat.shape)

# %%
