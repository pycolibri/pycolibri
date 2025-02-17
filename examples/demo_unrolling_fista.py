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

manual_device = "cpu"
# Check GPU support
print("GPU support: ", torch.cuda.is_available())


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
# Load dataset
# -----------------------------------------------
from colibri.data.datasets import CustomDataset

name = "cifar10"
path = "."
batch_size = 128

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
print('dir(models) ->', dir(models))  
from colibri import recovery


print('dir(recover) ->', dir(recovery.terms.prior))  
from colibri.recovery.fista import Fista
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.models.learned_proximals import LearnedPrior
from colibri.models.unrolling import UnrollingFISTA
from colibri.models.autoencoder import Autoencoder
import torch.nn as nn


algo_params = {
    "max_iters": 10,
    "alpha": 1e-4,
    "_lambda": 0.01,
}

fidelity = L2()
prior = Sparsity(basis="dct")
prior_args ={'in_channels': 3, 'out_channels': 3, 'feautures': [32, 64, 128, 256]}
models = nn.Sequential(*[Autoencoder(**prior_args) for _ in range(algo_params["max_iters"])])

print('number of parameters in models ->', sum(p.numel() for p in models.parameters() if p.requires_grad))
fista_unrolling = UnrollingFISTA(acquisition_model, fidelity, **algo_params, models=models).to(device)

# number of parameters in the model
num_params = sum(p.numel() for p in fista_unrolling.parameters() if p.requires_grad)
print(f"Number of parameters in the model: {num_params}")
# x0 = acquisition_model.forward(y, type_calculation="backward")

# x_hat = fista_unrolling(y, x0=x0)
#train the model on the dataset
# %%

epochs = 100
train_loader = data.DataLoader(dataset, batch_size=128, shuffle=True)
optimizer = torch.optim.Adam(fista_unrolling.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

for epoch in range(epochs):
    for i, data in enumerate(train_loader):
        input = data['input'].to(device)
        y = acquisition_model(input, type_calculation="forward")
        x0 = acquisition_model.forward(y, type_calculation="backward")

        output = fista_unrolling(y, x0=x0)
        optimizer.zero_grad()
        # output = fista_unrolling(input)
        loss = criterion(output, input)
        loss.backward()
        optimizer.step()

    if i % 1 == 0:
        print(f'Epoch {epoch}, Loss {loss.item()}')


# %%
# Visualize reconstruction
# -----------------------------------------------
plt.figure(figsize=(10, 10))
plt.imshow(output.squeeze().detach().cpu().permute(1, 2, 0))
plt.axis('off')
plt.title('Reconstructed image')
plt.show()









# print(x_hat.shape)

# %%
