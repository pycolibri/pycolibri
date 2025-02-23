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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# General imports
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

manual_device = False  # "cpu"
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


from colibri.models import build_network, Unet, Unet_KD

from colibri.misc import E2E
from colibri.train import Training
from colibri.train_kd import TrainingKD
from colibri.metrics import psnr, ssim

network_config = dict(
    in_channels=sample.shape[1],
    out_channels=sample.shape[1],
    reduce_spatial=True,  # Only for Autoencoder
)

recovery_model_student = build_network(Unet_KD, **network_config)

student = E2E(acquisition_model_student, recovery_model_student)
student = student.to(device)

optimizer_student = torch.optim.Adam(student.parameters(), lr=5e-4)

losses = {"MSE": torch.nn.MSELoss()}
metrics = {"PSNR": psnr, "SSIM": ssim}
losses_weights = [1.0]

n_epochs = 60
steps_per_epoch = None
frequency = 1

train_teacher = False
train_baseline = False

acquisition_model_teacher = SPC(
    input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=False
)


if train_teacher:
    recovery_model_teacher = build_network(Unet, **network_config)
    teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
    teacher = teacher.to(device)
    optimizer_teacher = torch.optim.Adam(teacher.parameters(), lr=5e-4)

    train_schedule = Training(
        model=teacher,
        train_loader=dataset_loader,
        optimizer=optimizer_teacher,
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
    torch.save(teacher.state_dict(), "teacher.pth")

elif not train_teacher:
    recovery_model_teacher = build_network(Unet_KD, **network_config)
    teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
    teacher = teacher.to(device)
    teacher.load_state_dict(torch.load("teacher.pth"))

if train_baseline:

    acquisition_model_baseline = SPC(
        input_shape=img_size, n_measurements=n_measurements, trainable=True, binary=True
    )

    recovery_model_baseline = build_network(Unet, **network_config)

    baseline = E2E(acquisition_model_baseline, recovery_model_baseline)
    baseline = baseline.to(device)

    optimizer_baseline = torch.optim.Adam(baseline.parameters(), lr=5e-4)

    train_schedule_baseline = Training(
        model=baseline,
        train_loader=dataset_loader,
        optimizer=optimizer_baseline,
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

    results_baseline = train_schedule_baseline.fit(
        n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, freq=frequency
    )
    torch.save(baseline.state_dict(), "baseline.pth")


train_schedule_kd = TrainingKD(
    student_model=student,
    teacher_model=teacher,
    teacher_path_weights="teacher.pth",
    train_loader=dataset_loader,
    optimizer=optimizer_student,
    loss_func={"MSE": torch.nn.MSELoss()},
    losses_weights=[0.1],
    kd_config={
        "enc_weight": 0.8,
        "dec_weight": 0.1,
        "loss_dec_type": "MSE",
        "loss_enc_type": "GRAMM",
        "layer_idxs": [3, 4],
        "att_config": None,
    },
    metrics=metrics,
    regularizers=None,
    regularizers_optics_mo=None,
    regularization_optics_weights_mo=None,
    regularizers_optics_ce=None,
    regularization_optics_weights_ce=None,
    regularization_weights=None,
    schedulers=[],
    callbacks=[],
    device=device,
)

results_kd = train_schedule_kd.fit(
    n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, freq=frequency
)
