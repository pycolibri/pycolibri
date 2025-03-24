r"""
Demo Knowledge distillation.
===================================================

In this example we show how to use a simple pipeline of knowledge distillation learning with the SD-CASSi system.

.. note::
    For more information check the following sections:
    
    * For 'E2E' details check :func:`colibri.misc.e2e.E2E`.
    * For 'KD' details check :func:`colibri.misc.kd.KD`.
    * For 'loss_dec_type' details check :func:`colibri.misc.kd.KD_enc_loss`.
    * For 'loss_enc_type' details check :func:`colibri.misc.kd.KD_dec_loss`.
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
from colibri.optics.functional import BinarizeSTE

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
acquisition_name = "cassi"  # ['spc', 'cassi', 'doe']


dataset = CustomDataset(name, path, train=True)


dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = next(iter(dataset_loader))["input"]
img = make_grid(sample[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10, 10))
plt.imshow(img.permute(2, 1, 0))
plt.title(f"{name} dataset")
plt.axis("off")
plt.show()

# %%
# Computational imaging systems configuration
# -----------------------------------------------
# Configuration of the computational imaging system for the teacher, student and baseline models.

import math
from colibri.optics import SD_CASSI

img_size = sample.shape[1:]

acquisition_model_student = SD_CASSI(input_shape=img_size, trainable=True, binary=True)

y = acquisition_model_student(sample)

img = make_grid(y[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10, 10))
plt.imshow(img.permute(2, 1, 0))
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
    features=[4, 8, 16, 32],
    last_activation="relu",
)

recovery_model_student = build_network(Unet_KD, **network_config)

student = E2E(acquisition_model_student, recovery_model_student)
student = student.to(device)

optimizer_student = torch.optim.AdamW(student.parameters(), lr=5e-4)
acquisition_model_teacher = SD_CASSI(input_shape=img_size, trainable=True, binary=False)


# %%
# Training configuration
# -----------------------------------------------
# We train the teacher model in an end-to-end manner, the student model with knowledge distillation, and the baseline model without knowledge distillation.

losses = {"MSE": torch.nn.MSELoss()}
metrics = {"PSNR": psnr, "SSIM": ssim}
losses_weights = [1.0]

n_epochs = 100
steps_per_epoch = None
frequency = 1

train_teacher = True
train_baseline = True

if train_teacher:
    print("Training teacher model")
    recovery_model_teacher = build_network(Unet, **network_config)
    teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
    teacher = teacher.to(device)
    optimizer_teacher = torch.optim.AdamW(teacher.parameters(), lr=5e-4)

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

    del teacher

    recovery_model_teacher = build_network(Unet_KD, **network_config)
    teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
    teacher = teacher.to(device)
    teacher.load_state_dict(torch.load("teacher.pth"))


elif not train_teacher:
    print("Loading teacher model")
    recovery_model_teacher = build_network(Unet_KD, **network_config)
    teacher = E2E(acquisition_model_teacher, recovery_model_teacher)
    teacher = teacher.to(device)
    teacher.load_state_dict(torch.load("teacher.pth"))

if train_baseline:
    print("Training baseline model")
    acquisition_model_baseline = SD_CASSI(input_shape=img_size, trainable=True, binary=True)

    recovery_model_baseline = build_network(Unet, **network_config)

    baseline = E2E(acquisition_model_baseline, recovery_model_baseline)
    baseline = baseline.to(device)

    optimizer_baseline = torch.optim.AdamW(baseline.parameters(), lr=5e-4)

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

elif not train_baseline:
    print("Loading baseline model")
    acquisition_model_baseline = SD_CASSI(input_shape=img_size, trainable=True, binary=True)

    recovery_model_baseline = build_network(Unet, **network_config)

    baseline = E2E(acquisition_model_baseline, recovery_model_baseline)
    baseline = baseline.to(device)
    baseline.load_state_dict(torch.load("baseline.pth"))

print("Training student model")
train_schedule_kd = TrainingKD(
    student_model=student,
    teacher_model=teacher,
    teacher_path_weights="teacher.pth",
    train_loader=dataset_loader,
    optimizer=optimizer_student,
    loss_func={"MSE": torch.nn.MSELoss()},
    losses_weights=[0.1],
    kd_config={
        "enc_weight": 0.9,
        "dec_weight": 0.0,
        "loss_dec_type": "MSE",
        "loss_enc_type": "GRAMM_SD_CASSI",
        "layer_idxs": [4],
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

# %%
# Plot results
# -----------------------------------------------
# We plot the results of the optimized optical elements for the teacher, student and baseline models.

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(
    acquisition_model_teacher.learnable_optics.cpu().detach().numpy().squeeze(),
    cmap="gray",
)
plt.title("Teacher CA")

plt.subplot(1, 3, 2)
plt.imshow(
    BinarizeSTE.apply(acquisition_model_student.learnable_optics.cpu().detach()).numpy().squeeze(),
    cmap="gray",
)
plt.title("Student CA")

plt.subplot(1, 3, 3)
plt.imshow(
    BinarizeSTE.apply(acquisition_model_baseline.learnable_optics.cpu().detach()).numpy().squeeze(),
    cmap="gray",
)
plt.title("Baseline CA")

plt.show()
