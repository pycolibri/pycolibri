r"""
Demo Colibri.
===================================================

In this example we show how to use a simple pipeline of knowledge distillation learning with the SPC system as teacher and.
"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(parent_dir)

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

name = "fashion_mnist"  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
path = "."
batch_size = 64
acquisition_name = "spc"  # ['spc', 'cassi', 'doe']


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
plt.title(f"{name} dataset")
plt.axis("off")
plt.show()

# %%
# Optics forward model
# -----------------------------------------------
# Define the forward operators :math:`\mathbf{y} = \mathbf{H}_\phi \mathbf{x}`, in this case, the CASSI and SPC forward models.
# Each optics model can comptute the forward and backward operators i.e., :math:`\mathbf{y} = \mathbf{H}_\phi \mathbf{x}` and :math:`\mathbf{x} = \mathbf{H}^T_\phi \mathbf{y}`.


import math
from colibri.optics import SPC

img_size = sample.shape[1:]

n_measurements = 256
n_measurements_sqrt = int(math.sqrt(n_measurements))

acquisition_model_teacher = SPC(
    input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=True
)
acquisition_model_student = SPC(
    input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=False
)
