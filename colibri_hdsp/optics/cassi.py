import torch
import numpy as np
from colibri_hdsp.optics.functional import forward_color_cassi, backward_color_cassi, forward_dd_cassi, backward_dd_cassi, forward_cassi, backward_cassi
class CASSI(torch.nn.Module):
    """
    Layer that performs the forward and backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763

    """

    def __init__(self, input_shape, mode, trainable=False, ca_regularizer=None, initial_ca=None, seed=None):
        """
        Args:
            mode (str): String, mode of the coded aperture, it can be "base", "dd" or "color"
            trainable (bool): Boolean, if True the coded aperture is trainable
            ca_regularizer (function): Regularizer function applied to the coded aperture
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, M, N, 1)
            seed (int): Random seed
        """
        super(CASSI, self).__init__()
        self.seed = seed
        self.trainable = trainable
        self.ca_regularizer = ca_regularizer
        self.initial_ca = initial_ca

        if mode == "base":
            self.direct = forward_cassi
            self.backward = backward_cassi
        elif mode == "dd":
            self.direct = forward_dd_cassi
            self.backward = backward_dd_cassi
        elif mode == "color":
            self.direct = forward_color_cassi
            self.backward = backward_color_cassi

        self.mode = mode
  
        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        if self.mode == 'base':
            shape = (1, 1, self.M, self.N)
        elif self.mode == 'dd':
            shape = (1, 1, self.M, self.N + self.L - 1)
        elif self.mode == 'color':
            shape = (1, self.L, self.M, self.N)
        else:
            raise ValueError(f"the mode {self.mode} is not valid")

        if self.initial_ca is None:
            initializer = torch.randn(shape, requires_grad=self.trainable)
        else:
            assert self.initial_ca.shape == shape, f"the start CA shape should be {shape} but is {self.initial_ca.shape}"
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        self.ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)

    def forward(self, x, type_calculation="forward"):
        """
        Call method of the layer, it performs the forward or backward operator according to the type_calculation
        Args:
            x (torch.Tensor): Input tensor with shape (1, M, N, L)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (1, M, N + L - 1, 1) if type_calculation is "forward", (1, M, N, L) if type_calculation is "backward, or (1, M, N, L) if type_calculation is "forward_backward
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """
        if type_calculation == "forward":
            return self.direct(x, self.ca)

        elif type_calculation == "backward":
            return self.backward(x, self.ca)
        elif type_calculation == "forward_backward":
            return self.backward(self.direct(x, self.ca), self.ca)

        else:
            raise ValueError("type_calculation must be forward, backward or forward_backward")
        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    import scipy.io as sio
    import os

    # load a mat file

    cube = sio.loadmat(os.path.join('examples', 'data', 'spectral_image.mat'))['img']  # (M, N, L)
    ca = np.random.rand(1, cube.shape[0], cube.shape[1], 1)  # custom ca (1, M, N, 1)

    # load optical encoder

    mode = 'base'
    device = 'cuda'
    cassi = CASSI(input_shape=cube.shape, mode=mode, device=device, trainable=False, initial_ca=ca)
    # encode the cube

    cube_tf = torch.from_numpy(cube).float().unsqueeze(0).to(device) # (1, M, N, L)
    measurement = cassi(cube_tf, type_calculation="forward")
    backward = cassi(measurement, type_calculation="backward")
    direct_backward = cassi(cube_tf)
    measurement2 = cassi(backward, type_calculation="forward_backward")

    # Print information about tensors

    print('cube shape: ', cube_tf.shape)
    print('measurement shape: ', measurement.shape)
    print('backward shape: ', backward.shape)

    # visualize the measurement

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.title('cube')
    plt.imshow(cube[..., 0])

    plt.subplot(222)
    plt.title('measurement')
    plt.imshow(measurement[0, ..., 0].cpu())

    plt.subplot(223)
    plt.title('backward')
    plt.imshow(backward[0, ..., 0].cpu())

    plt.subplot(224)
    plt.title('measurement2')
    plt.imshow(measurement2[0, ..., 0].cpu())

    plt.tight_layout()
    plt.show()
