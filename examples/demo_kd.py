r"""
Demo Colibri.
===================================================

In this example we show how to use a simple pipeline of knowledge distillation learning with the Colored CASSI system as teacher  and the SD-CASSI system as the student.
"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os

os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory ", os.getcwd())

# General imports
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

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

name = "cifar10"  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
path = "."
batch_size = 128
acquisition_name = "c_cassi"  # ['spc', 'cassi', 'doe']


dataset = CustomDataset(name, path)


dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = next(iter(dataset_loader))["input"]
img = make_grid(sample[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10, 10))
plt.imshow(img.permute(1, 2, 0))
plt.title("CIFAR10 dataset")
plt.axis("off")
plt.show()

# %%
# Optics forward model
# -----------------------------------------------
# Define the forward operators :math:`\mathbf{y} = \mathbf{H}_\phi \mathbf{x}`, in this case, the CASSI and SPC forward models.
# Each optics model can comptute the forward and backward operators i.e., :math:`\mathbf{y} = \mathbf{H}_\phi \mathbf{x}` and :math:`\mathbf{x} = \mathbf{H}^T_\phi \mathbf{y}`.


import math
from colibri.optics import SPC, SD_CASSI, DD_CASSI, C_CASSI, SingleDOESpectral
from colibri.optics.sota_does import spiral_doe, spiral_refractive_index

img_size = sample.shape[1:]

acquisition_config = dict(
    input_shape=img_size,
)


