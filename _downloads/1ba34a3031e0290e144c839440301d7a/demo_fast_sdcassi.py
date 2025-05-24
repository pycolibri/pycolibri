r"""
Demo Fast SD CASSI.
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


import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize

from colibri.data.datasets import CustomDataset
from colibri.optics import SD_CASSI
from colibri.recovery.solvers.sd_cassi import L2L2SolverSDCASSI


#General imports
import torch

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
# Parameters
# -----------------------------------------------
M = 256
N = M
L = 3
sample_index = 1

# %%
# Data
# -----------------------------------------------

dataset = CustomDataset('cave', path='data')
sample = dataset[sample_index]
x = sample['output'].numpy()
x = x[::-1, ...]
x = torch.from_numpy(resize(x[::int(x.shape[0] / L) + 1], [L, M, N]))[None, ...].float()

x = x / x.max()  # Normalize the image to [0, 1]

print(x.shape)

# %%
# Tenssor CASSI
# -----------------------------------------------

sd_cassi = SD_CASSI((L, M, N))
y = sd_cassi(x)

ca = sd_cassi.learnable_optics[0]



# %%
# Setup and run reconstruction
# -----------------------------------------------
recons_fn = L2L2SolverSDCASSI(y, sd_cassi)
xtilde    = torch.zeros_like(x, device=x.device) # initial guess
rho       = 1e-4  # regularization parameter

# solution with the solver
recons_img  = recons_fn.solve(xtilde, rho)

# solution with the manual inversion
[Qinv, rhoIQ] = recons_fn.ComputeQinv(rho)
recons_img2 = recons_fn.IMVMS(sd_cassi(y, type_calculation="backward"), Q=Qinv)

error = torch.norm(recons_img - recons_img2)  # compute the error

fig, ax = plt.subplots(2, 2, figsize=(6, 6))
plt.suptitle(f'Inversion error norm2(recon_img1 - recon_img2): {error:.4e}', fontsize=12)

ax[0, 0].imshow(x[0].permute(1, 2, 0) / x.max() )
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(y[0].permute(1, 2, 0))
ax[0, 1].set_title("Measurement")
ax[0, 1].axis("off")

ax[1, 0].imshow(recons_img[0].permute(1, 2, 0) / recons_img.max() )
ax[1, 0].set_title("Reconstructed Image 1")
ax[1, 0].axis("off")

ax[1, 1].imshow(recons_img2[0].permute(1, 2, 0) / recons_img2.max() )
ax[1, 1].set_title("Reconstructed Image 2")
ax[1, 1].axis("off")

plt.tight_layout()
plt.show()

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity
from colibri.recovery.pnp import PnP_ADMM


fidelity = L2()
prior = Sparsity(basis="dct")

algo_params = {
    'max_iters': 10,
    '_lambda': 0.1,
    'rho': 0.01,
    'alpha': 1e-4,
}

pnp = PnP_ADMM(sd_cassi, fidelity, prior, **algo_params)

x0 = recons_img
x_hat = pnp(y, x0=x0,  verbose=False).detach()


fig, ax = plt.subplots(1, 4, figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title("Measurement")
plt.imshow(y[0].permute(1, 2, 0))
plt.axis("off")

plt.subplot(1, 4, 2)
plt.title("Fast Inversion")
plt.imshow(recons_img[0].permute(1, 2, 0) / recons_img.max())
plt.axis("off")

plt.subplot(1, 4, 3)
plt.title("PnP ADMM")
plt.imshow(x_hat[0].permute(1, 2, 0) / x_hat.max())
plt.axis("off")

plt.subplot(1, 4, 4)
plt.title("Reference")
plt.imshow(x[0].permute(1, 2, 0) / x.max())
plt.axis("off")

plt.show()