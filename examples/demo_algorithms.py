# %%
# Select Working Directory and Device
# -----------------------------------------------
import os 
os.chdir(os.path.dirname(os.getcwd()))
print("Current Working Directory " , os.getcwd())

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
from colibri_hdsp.data.datasets import Dataset

dataset_path = 'cifar10'
keys = ''
batch_size = 1
dataset = Dataset(dataset_path, keys, batch_size)
adquistion_name = 'cassi' #  ['spc', 'cassi']




# %%
# Visualize dataset
# -----------------------------------------------
from torchvision.utils import make_grid
from colibri_hdsp.optimization.transforms import DCT2D

sample = next(iter(dataset.train_dataset))[0]


# %%


transform_dct = DCT2D()

theta = transform_dct.forward(sample)
x_hat = transform_dct.inverse(theta)

error = torch.norm(sample - x_hat)
print("Error: ", error  )


# %%
# Optics forward model

import math
from colibri_hdsp.optics import SPC, CASSI

img_size = sample.shape[1:]

acquisition_config = dict(
    input_shape = img_size,
)

if adquistion_name == 'spc':
    n_measurements  = 256 
    n_measurements_sqrt = int(math.sqrt(n_measurements))    
    acquisition_config['n_measurements'] = n_measurements

acquistion_model = {
    'spc': SPC(**acquisition_config),
    'cassi': CASSI(**acquisition_config),
}[adquistion_name]

y = acquistion_model(sample)

if adquistion_name == 'spc':
    y = y.reshape(y.shape[0], -1, n_measurements_sqrt, n_measurements_sqrt)





# Reconstruct image
from colibri_hdsp.optimization.algorithms.fista import Fista
from colibri_hdsp.optimization.prior import Sparsity
from colibri_hdsp.optimization.fidelity import L2
from colibri_hdsp.optimization.transforms import DCT2D

algo_params = {
    'max_iter': 2000,
    'alpha': 0.1,
    'lambda': 0.001,
    'tol': 1e-3
}



fidelity = L2()
prior = Sparsity()
transform = DCT2D()

fista = Fista(fidelity, prior, acquistion_model, algo_params, transform)

x0 = 0.01*torch.randn_like(theta) # + theta*0.1

x_hat = fista(y, x0=x0 ) 

print(x_hat.shape)
print("Error: ", torch.norm(sample - x_hat)  )



plt.figure(figsize=(10,10))

plt.subplot(1,4,1)
plt.imshow(sample[0,:,:].permute(1, 2, 0), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,2)
plt.imshow(abs(theta[0,:,:]).permute(1, 2, 0), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,3)
plt.imshow(y[0,:,:].permute(1, 2, 0), cmap='gray')
plt.xticks([])
plt.yticks([])

plt.subplot(1,4,4)
plt.imshow(x_hat[0,:,:].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
plt.xticks([])
plt.yticks([])