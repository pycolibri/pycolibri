r"""
Demo FISTA.
===================================================

"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os

from torch.utils import data

# os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory ", os.getcwd())

import sys

sys.path.append(os.path.join(os.getcwd()))

# General imports
import matplotlib.pyplot as plt
import torch
import os

# Set random seed for reproducibility
torch.manual_seed(0)

manual_device = False
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

acquisition_model = acquisition_model(**acquisition_config).to(device)

y = acquisition_model(sample)

# Reconstruct image
import colibri
from colibri import models
from colibri import recovery


from colibri.recovery.fista import Fista
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.models.learned_proximals import LearnedPrior
from colibri.models.unrolling import UnrollingFISTA
from colibri.models.autoencoder import Autoencoder



algo_params = {
    "max_iters": 20,
    "alpha": 1e-4,
    "_lambda": 0.01,
}

fidelity = L2()
prior_args ={'in_channels': 3, 'out_channels': 3, 'feautures': [32, 64, 128, 256]}
model =  torch.nn.ModuleList([Autoencoder(**prior_args).to(device) for i in range(algo_params['max_iters']) ])
prior = LearnedPrior(max_iter=2000, model=model, prior_args=prior_args).to(device)

fista_unrolling = UnrollingFISTA(acquisition_model, fidelity, **algo_params, model=model, prior_args=prior_args)

x0 = acquisition_model.forward(y, type_calculation="backward")

x_hat = fista_unrolling(y, x0=x0)




normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

# plt.figure(figsize=(10, 10))

# plt.subplot(1, 3, 1)
# plt.title("Reference")
# plt.imshow(sample[0, :, :].permute(1, 2, 0), cmap="gray")
# plt.xticks([])
# plt.yticks([])


# if acquisition_name == "spc":
#     y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)

# plt.subplot(1, 3, 2)
# plt.title("Measurement")
# plt.imshow(normalize(y[0, :, :]).permute(1, 2, 0), cmap="gray")
# plt.xticks([])
# plt.yticks([])

# plt.subplot(1, 3, 3)
# plt.title("Reconstruction")
# plt.imshow(normalize(x_hat[0, :, :]).permute(1, 2, 0).detach().cpu().numpy(), cmap="gray")
# plt.xticks([])
# plt.yticks([])

# plt.show()

# %%
# train the model
# -----------------------------------------------

optimizer = torch.optim.Adam(fista_unrolling.parameters(), lr=1e-3)

losses = {"MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss()}

metrics = {"PSNR": colibri.metrics.psnr, "SSIM": colibri.metrics.ssim}

losses_weights = [1.0, 1.0]

n_epochs = 10

steps_per_epoch = 10

for data in enumerate(dataset):
    
    sample = data[1]["input"]
    sample = sample.unsqueeze(0).to(device)
    
    y = acquisition_model(sample)
    
    x0 = acquisition_model.forward(y, type_calculation="backward")
    
    x_hat = fista_unrolling(y, x0=x0)
    
    loss = sum(
        [
            weight * loss_fn(x_hat, sample)
            for weight, loss_fn in zip(losses_weights, losses.values())
        ]
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
 