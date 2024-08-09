r"""
Demo Algorithms.
===================================================

"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os 
os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory " , os.getcwd())

import sys
sys.path.append(os.path.join(os.getcwd()))

#General imports
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

name = 'cifar10'
path = '.'
batch_size = 1

builtin_dict = dict(train=True, download=True)
dataset = CustomDataset(name, path,
                        builtin_dict=builtin_dict,
                        transform_dict=None)
acquisition_name = 'spc' #  ['spc', 'cassi']




# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid
from colibri.recovery.terms.transforms import DCT2D

def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

sample = dataset[0]['input']
sample = sample.unsqueeze(0).to(device)


# %%
# Optics forward model

import math
from colibri.optics import SPC, SD_CASSI, DD_CASSI, C_CASSI

img_size = sample.shape[1:]

acquisition_config = dict(
    input_shape = img_size,
)

if acquisition_name == 'spc':
    n_measurements  = 25**2 
    n_measurements_sqrt = int(math.sqrt(n_measurements))    
    acquisition_config['n_measurements'] = n_measurements

acquisition_model = {
    'spc': SPC,
    'sd_cassi': SD_CASSI,
    'dd_cassi': DD_CASSI,
    'c_cassi': C_CASSI
}[acquisition_name]

acquisition_model = acquisition_model(**acquisition_config)

y = acquisition_model(sample)

# Reconstruct image
from colibri.recovery.pnp import PnP_ADMM
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.transforms import DCT2D

algo_params = {
    'max_iters': 200,
    '_lambda': 0.05,
    'rho': 0.1,
    'alpha': 1e-4,
}



fidelity  = L2()
prior     = Sparsity(basis='dct')

pnp = PnP_ADMM(acquisition_model, fidelity, prior, **algo_params)

x0 = acquisition_model.forward(y, type_calculation="backward")
x_hat = pnp(y, x0=x0,  verbose=True) 

basis = DCT2D()
theta = basis.forward(x_hat).detach()

print(x_hat.shape)

plt.figure(figsize=(10,10))

plt.subplot(1,4,1)
plt.title('Reference')
plt.imshow(normalize(sample[0,:,:].permute(1, 2, 0)), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,2)
plt.title('Sparse Representation')
plt.imshow(normalize(abs(theta[0,:,:]).permute(1, 2, 0)), cmap='gray')
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
x_hat -= x_hat.min()
x_hat /= x_hat.max()
plt.imshow(normalize(x_hat[0,:,:].permute(1, 2, 0).detach().cpu().numpy()), cmap='gray')
plt.xticks([])
plt.yticks([])
