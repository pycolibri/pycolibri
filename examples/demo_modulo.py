r"""
Demo Modulo.
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
# Load libraries and data
# -----------------------------------------------
from PIL import Image
import requests
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

from colibri.optics import Modulo
from colibri.recovery.solvers import L2L2SolverModulo
import torchvision.transforms as transforms

np2tensor = lambda x: torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float()

def channel_norm(x):
    """ Normalize the input tensor to have values between 0 and 1
    Args:
        x (torch.Tensor): Input tensor with shape (B, C, H, W)
    """
    x -= x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    x /= x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    return x


url = "https://optipng.sourceforge.net/pngtech/corpus/kodak/kodim23.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
img = np.array(img) / (255 + 1e-3)
img = np2tensor(img)
# Guarantee that the image has values between 0 and 1 by channel
img = channel_norm(img)

img_size = 512
saturation_factor = 1.5
blur_fn = transforms.GaussianBlur(7, sigma=(1, 1))

img = torch.nn.functional.interpolate(img, size=img_size) * saturation_factor
# apply blur
img = blur_fn(img)


modulo_sensing = Modulo()
mod_img = modulo_sensing(img)

# %%
# Setup and run reconstruction
# -----------------------------------------------
recons_fn = L2L2SolverModulo(img, modulo_sensing)
xtilde    = None # initial guess
rho       = 0.0  # regularization parameter

recons_img  = recons_fn.solve(xtilde, rho)
# Since the DCT solver returns a tensor with 0 mean
# we need to normalize the image to have values between 0 and 1
recons_img  = channel_norm(recons_img)


fig, ax = plt.subplots(1, 3, figsize=(10, 10))


ax[0].imshow(img[0].permute(1, 2, 0) / saturation_factor)
ax[0].set_title("Original Image")
ax[0].axis("off")

ax[1].imshow(mod_img[0].permute(1, 2, 0))
ax[1].set_title("Modulo Image")
ax[1].axis("off")

ax[2].imshow(recons_img[0].permute(1, 2, 0))
ax[2].set_title("Reconstructed Image")
ax[2].axis("off")

plt.show()
