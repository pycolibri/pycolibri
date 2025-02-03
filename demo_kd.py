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

name = "cifar10"  # ['cifar10', 'cifar100', 'mnist', 'fashion_mnist', 'cave']
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
    input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=False
)
acquisition_model_student = SPC(
    input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=True
)

y = acquisition_model_student(sample)

if acquisition_name == "spc":
    y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)

img = make_grid(y[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10, 10))
plt.imshow(img.permute(1, 2, 0))
plt.axis("off")
plt.title(f"{acquisition_name.upper()} measurements")
plt.show()


from colibri.models import build_network, Unet
from colibri.misc import E2E
from colibri.train import Training
from colibri.metrics import psnr, ssim

network_config = dict(
    in_channels=sample.shape[1],
    out_channels=sample.shape[1],
    reduce_spatial=True,  # Only for Autoencoder
)

recovery_model_teacher = build_network(Unet, **network_config)

teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
optimizer = torch.optim.Adam(teacher.parameters(), lr=5e-4)
losses = {"MSE": torch.nn.MSELoss()}
metrics = {"PSNR": psnr, "SSIM": ssim}
losses_weights = [1.0, 1.0]

n_epochs = 50
steps_per_epoch = None
frequency = 1

train_schedule = Training(
    model=teacher,
    train_loader=dataset_loader,
    optimizer=optimizer,
    loss_func=losses,
    losses_weights=losses_weights,
    metrics=metrics,
    regularizers=None,
    regularization_weights=None,
    schedulers=[],
    callbacks=[],
    device=device,
    regularizers_optics_ce=None,
    regularization_optics_weights_ce=None,
    regularizers_optics_mo=None,
    regularization_optics_weights_mo=None,
)

results = train_schedule.fit(n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, freq=frequency)
