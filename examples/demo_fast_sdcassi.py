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
L = 1
sample_index = 1

# %%
# Data
# -----------------------------------------------

dataset = CustomDataset('cave', path='data')
sample = dataset[sample_index]
x = sample['output'].numpy()
x = torch.from_numpy(resize(x[::int(x.shape[0] / L) + 1], [L, M, N]))[None, ...].float()

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
rho       = 1.0  # regularization parameter

# solution with the solver
# recons_img  = recons_fn.solve(xtilde, rho)
recons_img = y / ca

# solution with the manual inversion
[Qinv, rhoIQ] = recons_fn.ComputeQinv(rho)
recons_img2 = recons_fn.IMVMS(sd_cassi(y, type_calculation="backward"), Q=Qinv)

error = torch.norm(recons_img - recons_img2)  # compute the error

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
plt.suptitle(f'Inversion error norm2(recon_img1 - recon_img2): {error:.4e}', fontsize=16)

ax[0, 0].imshow(x[0, 0] / x.max() )
ax[0, 0].set_title("Original Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(y[0, 0])
ax[0, 1].set_title("Measurement")
ax[0, 1].axis("off")

ax[1, 0].imshow(recons_img[0, 0] / recons_img.max() )
ax[1, 0].set_title("Reconstructed Image 1")
ax[1, 0].axis("off")

ax[1, 1].imshow(recons_img2[0, 0]/ recons_img2.max() )
ax[1, 1].set_title("Reconstructed Image 2")
ax[1, 1].axis("off")

plt.tight_layout()
plt.show()
