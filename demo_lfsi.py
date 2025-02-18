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
import torchvision
name = "cifar10"
path = "."
batch_size = 1


dataset = CustomDataset(name, path)


# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = dataset[0]["input"]
sample = sample.mean(0)
sample = sample.unsqueeze(0).to(device)
sample = torchvision.transforms.Resize((128, 128))(sample)
sample = 1*(sample - torch.min(sample)) / (torch.max(sample) - torch.min(sample))
sample = torch.exp(1j * 2*torch.pi * sample)
sample = torch.nn.functional.pad(sample, (32, 32, 32, 32), mode='constant', value=0)
# %%
# Optics forward model

from colibri.optics import CodedPhaseImaging
from colibri.optics.functional import coded_phase_imaging_forward, coded_phase_imaging_backward


img_size = sample.shape[1:]



wave_length = 670e-9
pixel_size = 1e-6
sensor_distance = 50e-6
approximation = "fresnel"



acquisition_model = CodedPhaseImaging(
    input_shape=img_size,
    pixel_size=pixel_size,
    wavelength=wave_length,
    sensor_distance=sensor_distance,
    approximation=approximation,
    trainable=False,
)


y = acquisition_model(sample, type_calculation="forward", intensity=True)


# Reconstruct image
from colibri.recovery import FilteredSpectralInitialization


lfsi_algorithm = FilteredSpectralInitialization(
                    max_iterations=15,
                    p=0.9,
                    k_size=5,
                    sigma=1.0,
                    train_filter=False,
                    dtype=torch.float32,
                    device=device,
)

x_hat = lfsi_algorithm(y, acquisition_model)


sample = sample.detach().cpu().squeeze().angle()
y = y.detach().cpu().squeeze()
x_hat = x_hat.detach().cpu().squeeze().angle()

normalize = lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].set_title("Reference")
axs[0].imshow(sample, cmap="gray")
axs[0].set_xticks([])
axs[0].set_yticks([])

axs[1].set_title("Measurement")
axs[1].imshow(y, cmap="gray")
axs[1].set_xticks([])
axs[1].set_yticks([])

axs[2].set_title("Estimation")
axs[2].imshow(x_hat, cmap="gray")
axs[2].set_xticks([])
axs[2].set_yticks([])

plt.show()

# %%
