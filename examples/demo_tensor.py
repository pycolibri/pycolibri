r"""
Demo Tensor.
===================================================

This is a pytorch implementation of paper "Fast matrix inversion in compressive spectral imaging based on a tensorial representation"
https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-33/issue-1/013034/Fast-matrix-inversion-in-compressive-spectral-imaging-based-on-a/10.1117/1.JEI.33.1.013034.short#_=_
"""

# %%
# Select Working Directory and Device
# -----------------------------------------------
import os
import sys

from skimage.transform import resize

from colibri.data.datasets import CustomDataset
from colibri.optics.tensor_cassi import TensorCASSI

sys.path.append(os.getcwd())
print("Current Working Directory ", os.getcwd())

# Libraries
import scipy.io as sio
import matplotlib.pyplot as plt

import torch

torch.set_default_dtype(torch.float64)

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
sample_index = 1

# %%
# Data
# -----------------------------------------------

dataset = CustomDataset('cave', path='data')
sample = dataset[sample_index]
x = sample['output'].numpy()
x = torch.from_numpy(resize(x[::int(x.shape[0] / L) + 1], [L, M, N]))[None, ...]

plt.figure()
plt.imshow(x[0, 0])
plt.title('Spectral Image')
plt.show()

# %%
# Tenssor CASSI
# -----------------------------------------------

tensor_cassi = TensorCASSI((L, M, N), mode="base", trainable=False)

# %%
# Sensing and computation of P and Q
# -----------------------------------------------

y = tensor_cassi(x)
P = tensor_cassi.P
Q = tensor_cassi.Q

plt.figure()
plt.imshow(y[0, 0])
plt.title('Measurement')

plt.figure()
plt.imshow(P[0, 0])
plt.title('$P_{11}$')

plt.figure()
plt.imshow(Q[0, 0])
plt.title('$Q_{11}$')
plt.show()

# %%
# We now check HH*y or Pb
# -----------------------------------------------

HHstarb1 = tensor_cassi(tensor_cassi(y, type_calculation='backward'))
HHstarb2 = tensor_cassi.IMVM(y)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(HHstarb1[0, 0]), plt.title('$HH^{*}$   (1)')
plt.subplot(122), plt.imshow(HHstarb2[0, 0]), plt.title('$HH^{*}$   (2)')
plt.show()

error = torch.norm(HHstarb1 - HHstarb2).item()
print('$HH*$ error:', error)

# %%
# We now check H*Hb or Qd
# -----------------------------------------------

HstarHd1 = tensor_cassi(tensor_cassi(x), type_calculation='backward')
HstarHd2 = tensor_cassi.IMVMS(x)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(HstarHd1[0, 0]), plt.title('$H^{*}H$   (1)')
plt.subplot(122), plt.imshow(HstarHd2[0, 0]), plt.title('$H^{*}H$   (2)')
plt.show()

error = torch.norm(HstarHd1 - HstarHd2).item()
print('$H*H$ error:', error)

# %%
# We now check (rhoI + HH*)^{-1} or (rhoI + P)^{-1}
# InvertedMeasurement: (rhoI + HH*)^{-1}(rhoI + HH*)y = y
# -----------------------------------------------

rho = 1
[Pinv, rhoIP] = tensor_cassi.ComputePinv(rho)
InvertedMeasurement = tensor_cassi.IMVM(rho * y + HHstarb1, P=Pinv)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(P[0, 0]), plt.title('$(I+P)_{11}^{-1}$')
plt.subplot(122), plt.imshow(InvertedMeasurement[0, 0]), plt.title('Inverted Measurement')
plt.show()

error = torch.norm(y - InvertedMeasurement).item()
print(f'Inversion error: {error}')

# %%
# We now check (rhoI + H*H)^{-1} or (rhoI + Q)^{-1}
# InvertedMeasurement: (rhoI + HH*)^{-1}(rhoI + HH*)y = y
# -----------------------------------------------

rho = 1
[Qinv, rhoIQ] = tensor_cassi.ComputeQinv(rho)
InvertedImage = tensor_cassi.IMVMS(rho * x + HstarHd1, Q=Qinv)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(Q[0, 0]), plt.title('$(I+Q)_{11}^{-1}$')
plt.subplot(122), plt.imshow(InvertedImage[0, 0]), plt.title('Inverted Image')
plt.show()

error = torch.norm(x - InvertedImage).item()
print(f'Inversion error: {error}')

# %%
# Employing the Woodbury Lemma
# -----------------------------------------------

X = tensor_cassi(y, type_calculation='backward')
invHtH1 = X - tensor_cassi(tensor_cassi.IMVM(tensor_cassi(X), P=Pinv), type_calculation='backward')
invHtH2 = tensor_cassi.IMVMS(X, Q=Qinv)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(invHtH1[0, 0]), plt.title('$E-H*(I+P)_{11}^{-1}Hy$')
plt.subplot(122), plt.imshow(invHtH2[0, 0]), plt.title('$(I+Q)_{11}^{-1}y$')
plt.show()

error = torch.norm(invHtH1 - invHtH2)
print(f'Inversion error: {error}')
