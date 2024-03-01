r"""
Demo Tensor.
===================================================

This is a pytorch implementation of paper "Fast matrix inversion in compressive spectral imaging based on a tensorial representation"

"""



# %%
# Select Working Directory and Device
# -----------------------------------------------
import os
import sys
sys.path.append(os.getcwd())
print("Current Working Directory " , os.getcwd())

# Libraries
import numpy as np
import scipy.io as sio
from skimage.transform import resize
import matplotlib.pyplot as plt

import torch
from colibri_hdsp.optics.tensorial import *

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
M = 32
N = M
L = 4
S = 2

# %%
# Data
# -----------------------------------------------
d = sio.loadmat('examples/data/lego.mat')['hyperimg']
d = resize(d[..., ::int(d.shape[-1] / L)], [M, N, L])
d = torch.tensor(d, dtype=torch.float32).permute(2, 0, 1)

plt.figure()
plt.imshow(d[0])
plt.title('Spectral Image')
plt.show()

# %%
# Coded Apertures
# -----------------------------------------------
CAs = torch.zeros((S, M, N))
for s in range(S):
    CAs[s] = torch.round(torch.rand(M, N))

# %%
# Sensing and computation of P and Q
# -----------------------------------------------

b = sensingH(d, CAs)
P = computeP(CAs, L)
Q = computeQ(CAs, L)

plt.figure()
plt.imshow(b[0])
plt.title('Measurement')

plt.figure()
plt.imshow(P[0, 0])
plt.title('P_11')

plt.figure()
plt.imshow(Q[0, 0])
plt.title('Q_11')

# %%
# We now check formula (5) for HH*b or Pb
# -----------------------------------------------
HHstarb1 = sensingH(sensingHt(b, CAs), CAs)
HHstarb2 = IMVM(P, b)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(HHstarb1[0]), plt.title('HH*   (1)')
plt.subplot(122), plt.imshow(HHstarb2[0]), plt.title('HH*   (2)')

error = torch.norm(HHstarb1 - HHstarb2)
print(f'HH* accuracy: {error}')
