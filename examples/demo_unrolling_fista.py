r"""
Demo Unrolling FISTA.
===================================================
In this example, we show how to train an Unrolling FISTA network following:

.. math::

    \arg\min_{\theta} \sum_{p=1}^P \left\| \mathcal{N}_{\theta^K} \left( \mathcal{N}_{\theta^{K-1}} \left( \cdots \mathcal{N}_{\theta^1} \left( \forwardLinear_\learnedOptics(\mathbf{x}_p) \right) \right) \right) \right\|_2

where :math:`\mathcal{N}_{\theta^k}, k = 1,\dots K` are the stages of the unrolling network.
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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os

# Set random seed for reproducibility
torch.manual_seed(0)

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Load dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset


name = 'fashion_mnist'  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
path = 'data'
batch_size = 128
builtin_train = True
builtin_download = True

dataset = CustomDataset(name, path, builtin_train, builtin_download)
dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

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
    n_measurements = 10**2
    n_measurements_sqrt = int(math.sqrt(n_measurements))
    acquisition_config["n_measurements"] = n_measurements

acquisition_model = {"spc": SPC, "sd_cassi": SD_CASSI, "dd_cassi": DD_CASSI, "c_cassi": C_CASSI}[
    acquisition_name
]

acquisition_model = acquisition_model(**acquisition_config).to(device)

y = acquisition_model(sample)

# Reconstruct image
from colibri import models

from colibri.recovery.terms.fidelity import L2
from colibri.models.unrolling import UnrollingFISTA
from colibri.models.learned_proximals import SparseProximalMapping
import torch.nn as nn

# %%
# FISTA Unrolling algorithm with K stages using SparseProximalMapping prior 


algo_params = {
    "max_iters": 10,
    "alpha": 1e-4,
    "_lambda": 0.01,
}

fidelity = L2()
prior_args ={'autoencoder_args': {'in_channels': 1, 'out_channels': 1, 'feautures': [32,64,128,256]},'beta': 1e-3}
models = nn.Sequential(*[SparseProximalMapping(**prior_args) for _ in range(algo_params["max_iters"])])

print('number of parameters in models ->', sum(p.numel() for p in models.parameters() if p.requires_grad))
fista_unrolling = UnrollingFISTA(acquisition_model, fidelity, **algo_params, models=models).to(device)

# [TODO] Create fista_unrolling.load_state_dict(torch.load("fista_unrolling.pth"))

# number of parameters in the model
num_params = sum(p.numel() for p in fista_unrolling.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params}")


# %%
# Training for only one epoch and 100 minibatches
from colibri.metrics import psnr, ssim, mse, mae

epochs = 100
optimizer = torch.optim.Adam(fista_unrolling.parameters(), lr=5e-4)
criterion = torch.nn.MSELoss()
input = next(iter(dataset_loader))["input"].to(device)
y = acquisition_model(input, type_calculation="forward")
x0 = acquisition_model.forward(y, type_calculation="backward")
output = fista_unrolling(y, x0=x0)


psnr_value = psnr(output, input)
ssim_value = ssim(output, input)


print(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}")




def normalize(image):
    return (image - image.min()) / (image.max() - image.min())



# %%
# Visualize reconstruction
# -----------------------------------------------

input = input.cpu().detach()
x0 = x0.cpu().detach()
y = y.cpu().detach()
output = output.cpu().detach()
plt.figure(figsize=(10,10))

plt.subplot(1,4,1)
plt.title('Reference')
plt.imshow(normalize(input[0,:,:].permute(1, 2, 0)), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,2)
plt.title('Intilization')
plt.imshow(normalize(abs(x0[0,:,:]).permute(1, 2, 0)), cmap='gray')
plt.xticks([])
plt.yticks([])


if acquisition_name == 'spc':
    y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)


plt.subplot(1,4,3)
plt.title('Measurement')
plt.imshow(normalize(y[0,:,:].permute(1, 2, 0)), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,4)
plt.title('Reconstruction')

plt.imshow(normalize(output[0,:,:].permute(1, 2, 0).detach().cpu().numpy()), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.show()

