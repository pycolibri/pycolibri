r"""
Demo Colibri.
===================================================

In this example we show how to use a simple pipeline of end-to-end learning with the CASSI and SPC forward models. Mainly, the forward model is defined,

.. math::
    \mathbf{y} = \mathbf{H}_\phi \mathbf{x}

where :math:`\mathbf{H}` is the forward model, :math:`\mathbf{x}` is the input image and :math:`\mathbf{y}` is the measurement and :math:`\phi` are the coding elements of the forward model. The recovery model is defined as,

.. math::
    \mathbf{x} = \mathcal{G}_\theta( \mathbf{y})

where :math:`\mathcal{G}` is the recovery model and :math:`\theta` are the parameters of the recovery model.

The training is performed by minimizing the following loss function,

.. math::
    \{\phi^*,\theta^*\} = \arg \min_{\phi,\theta} \sum_{p=1}^{P}\mathcal{L}(\mathbf{x}_p, \mathcal{G}_\theta( \mathbf{H}_\phi \mathbf{x}_p)) + \lambda \mathcal{R}(\phi) + \mu \mathcal{R}(\mathbf{H}_\phi \mathbf{x}) 

where :math:`\mathcal{L}` is the loss function, :math:`\mathcal{R}` is the regularizer, :math:`\lambda` and :math:`\mu` are the regularization weights, and :math:`P` is the number of samples in the training dataset.

"""



# %%
# Select Working Directory and Device
# -----------------------------------------------
import os, sys
os.chdir(os.path.dirname(os.getcwd()))
#os.chdir('colibri')
#sys.path.append(os.path.dirname(os.getcwd()))
print("Current Working Directory " , os.getcwd())

#General imports
import matplotlib.pyplot as plt
import torch
import os

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
from colibri.data.datasets import Dataset

dataset_path = 'cifar10'
keys = ''
batch_size = 128
dataset = Dataset(dataset_path, keys, batch_size)
acquisition_name = 'c_cassi' #  ['spc', 'cassi', 'doe']


# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid

sample = next(iter(dataset.train_dataset))[0]
img = make_grid(sample[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10,10))
plt.imshow(img.permute(1, 2, 0))
plt.title('CIFAR10 dataset')
plt.axis('off')
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
    input_shape = img_size,
)

if acquisition_name == 'spc':
    n_measurements  = 256 
    n_measurements_sqrt = int(math.sqrt(n_measurements))    
    acquisition_config['n_measurements'] = n_measurements

elif acquisition_name == 'doe':
    wavelengths = torch.Tensor([450, 550, 650])*1e-9
    doe_size = [100, 100]
    radius_doe = 0.5e-3
    source_distance = 1# meters
    sensor_distance=50e-3
    pixel_size = (2*radius_doe)/min(doe_size)
    height_map, aperture = spiral_doe(M = doe_size[0], N = doe_size[1], 
                    number_spirals = 3, radius = radius_doe, 
                    focal = 50e-3, start_w = 450e-9, end_w = 650e-9)
    refractive_index = spiral_refractive_index

    acquisition_config.update({"height_map":height_map, 
                        "aperture":aperture, 
                        "wavelengths":wavelengths, 
                        "source_distance":source_distance, 
                        "sensor_distance":sensor_distance, 
                        "sensor_spectral_sensitivity":lambda x: x,
                        "pixel_size":pixel_size,
                        "doe_refractive_index":refractive_index,
                        "trainable":True})

acquisition_model = {
    'spc': SPC,
    'sd_cassi': SD_CASSI,
    'dd_cassi': DD_CASSI,
    'c_cassi': C_CASSI,
    'doe': SingleDOESpectral,
}[acquisition_name]

acquisition_model = acquisition_model(**acquisition_config)

y = acquisition_model(sample)

if acquisition_name == 'spc':
    y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)

img = make_grid(y[:32], nrow=8, padding=1, normalize=True, scale_each=False, pad_value=0)

plt.figure(figsize=(10,10))
plt.imshow(img.permute(1, 2, 0))
plt.axis('off')
plt.title(f'{acquisition_name.upper()} measurements')
plt.show()


# %%
# Reconstruction model
# -----------------------------------------------
# Define the recovery model :math:`\mathbf{x} = \mathcal{G}_\theta( \mathbf{y})`, in this case, a simple U-Net model. 
# You can add you custom model by using the :meth: `build_network` function.
# Additionally we define the end-to-end model that combines the forward and recovery models.
# Define the loss function :math:`\mathcal{L}`, and the regularizers :math:`\mathcal{R}` for the forward and recovery models. 


from colibri.models import build_network, Unet, Autoencoder
from colibri.misc import E2E
from colibri.train import Training
from colibri.metrics import psnr, ssim


from colibri.regularizers import (
    Reg_Binary,
    Reg_Transmittance,
    MinVariance,
    KLGaussian,
)


network_config = dict(
    in_channels=sample.shape[1],
    out_channels=sample.shape[1],
    reduce_spatial = True           # Only for Autoencoder
)

recovery_model = build_network(Unet, **network_config)

model = E2E(acquisition_model, recovery_model)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = {"MSE": torch.nn.MSELoss(), "L1": torch.nn.L1Loss()}
metrics = {"PSNR": psnr, "SSIM": ssim}
losses_weights = [1.0, 1.0]

n_epochs = 10
steps_per_epoch = 10
frequency = 1

if "cassi" in acquisition_name or "spc" in acquisition_name:
    regularizers_optics_ce = {"RB": Reg_Binary(), "RT": Reg_Transmittance()}
    regularizers_optics_ce_weights = [50, 1]
else:
    regularizers_optics_ce = {}
    regularizers_optics_ce_weights = []


regularizers_optics_mo = {"MV": MinVariance(), "KLG": KLGaussian(stddev=0.1)}
regularizers_optics_mo_weights = [1e-3, 0.1]

train_schedule = Training(
    model=model,
    train_loader=dataset.train_dataset,
    optimizer=optimizer,
    loss_func=losses,
    losses_weights=losses_weights,
    metrics=metrics,
    regularizers=None,
    regularization_weights=None,
    schedulers=[],
    callbacks=[],
    device=device,
    regularizers_optics_ce=regularizers_optics_ce,
    regularization_optics_weights_ce=regularizers_optics_ce_weights,
    regularizers_optics_mo=regularizers_optics_mo,
    regularization_optics_weights_mo=regularizers_optics_mo_weights,
)

results = train_schedule.fit(
    n_epochs=n_epochs, steps_per_epoch=steps_per_epoch, freq=frequency
)


# %%
# Visualize results
# -----------------------------------------------
# Performs the inference :math:`\tilde{\mathbf{x}} = \mathcal{G}_{\theta^*}( \mathbf{H}_{\phi^*}\mathbf{x})` and visualize the results.

x_est = model(sample.to(device)).cpu()
y = acquisition_model(sample.to(device)).cpu()

if acquisition_name == 'spc':
    y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)

img      = make_grid(sample[:16], nrow=4, padding=1, normalize=True, scale_each=False, pad_value=0)
img_est  = make_grid(x_est[:16], nrow=4, padding=1, normalize=True, scale_each=False, pad_value=0)
img_y    = make_grid(y[:16], nrow=4, padding=1, normalize=True, scale_each=False, pad_value=0)

imgs_dict = {
    "CIFAR10 dataset": img, 
    f"{acquisition_name.upper()} measurements": img_y,
    "Recons CIFAR10": img_est
}

plt.figure(figsize=(14, 2.7))

for i, (title, img) in enumerate(imgs_dict.items()):
    plt.subplot(1, 4, i+1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')

ca = acquisition_model.learnable_optics.cpu().detach().numpy().squeeze()
if acquisition_name == 'spc':
    ca = ca = ca.reshape(n_measurements, 32, 32, 1)[0]
elif acquisition_name == 'c_cassi':
    ca = ca.transpose(1, 2, 0)
plt.subplot(1, 4, 4)
plt.imshow(ca, cmap='gray')
plt.axis('off')
plt.title('Learned CA')
plt.colorbar()

plt.show()
