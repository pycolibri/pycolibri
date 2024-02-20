import torch
import torch.nn as nn
from .functional import forward_spc, backward_spc


class SPC(nn.Module):

    def __init__(self, input_shape, n_measurements=256, trainable=False, initial_ca=None):
        """
        Initializes the Single Pixel Camera (SPC) model.

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            n_measurements (int): Number of measurements.
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, M, N, 1)
        """
        super(SPC, self).__init__()
        _, M, N = input_shape
        self.trainable = trainable
        self.initial_ca = initial_ca
        if self.initial_ca is None:
            initializer = torch.randn((n_measurements, M*N), requires_grad=self.trainable)
        else:
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        self.ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)


    def forward(self, x, type_calculation="forward"):
        """
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Output tensor after measurement.
        """
        if type_calculation == "forward":
            return forward_spc(x, self.ca)
        elif type_calculation == "backward":
            return backward_spc(x, self.ca)
        elif type_calculation == "forward_backward":
            return backward_spc(forward_spc(x, self.ca), self.ca)
        
        else:
            raise ValueError("type_calculation must be 'forward', 'backward' or 'forward_backward'")
        
        
    def ca_reg(self,reg):
        """
        Regularization of the coded aperture.

        Args:
            reg (function): Regularization function.
        
        Returns:
            torch.Tensor: Regularization value.
        """
        reg_value = reg(self.ca)
        return reg_value

    def measurements_reg(self,reg,x):
        """
        Regularization of the measurements.

        Args:
            reg (function): Regularization function.
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Regularization value.
        """
        y = self(x, type_calculation="forward")
        reg_value = reg(y)
        return reg_value
