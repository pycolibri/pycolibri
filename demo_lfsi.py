r"""
Demo LFSI.
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


dataset = CustomDataset(name, path)




# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = dataset[0]["input"]
sample = sample.unsqueeze(0).to(device)

# %%
# Optics forward model

import math
from colibri.optics import CodedPhaseImaging


img_size = sample.shape[1:]

acquisition_config = dict(
    input_shape=img_size,
)


wave_length = 550e-9
pixel_size = 1e-6
sensor_distance = 0.1
approximation = "fresnel"



acquisition_model = CodedPhaseImaging(
    input_shape=img_size,
    pixel_size=pixel_size,
    wavelength=torch.tensor([wave_length]),
    sensor_distance=sensor_distance,
    approximation=approximation,
    trainable=False,
)

sample = (sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
sample = torch.exp(1j * 2*torch.pi * sample)
y = acquisition_model(sample, type_calculation="forward", intensity=True)

# Reconstruct image
from colibri.recovery import FilteredSpectralInitialization


lfsi_algorithm = FilteredSpectralInitialization(
                    input_shape=y.shape,
                    max_iterations=15,
                    p=0.6,
                    k_size=5,
                    sigma=1.0,
                    train_filter=False,
                    dtype=torch.float32,
                    device=device,
)

x_hat = lfsi_algorithm(y, acquisition_model)


sample = sample.detach().cpu().squeeze().angle()
y = y.detach().cpu().squeeze().angle()
x_hat = x_hat.detach().cpu().squeeze().angle()

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

plt.figure(figsize=(10, 10))

plt.subplot(1, 4, 1)
plt.title("Reference")
plt.imshow(sample[:, :], cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 3)
plt.title("Measurement")
plt.imshow(normalize(y[:, :]), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 4, 4)
plt.title("Reconstruction")
plt.imshow(normalize(x_hat[:, :]), cmap="gray")
plt.xticks([])
plt.yticks([])

plt.show()

# %%
