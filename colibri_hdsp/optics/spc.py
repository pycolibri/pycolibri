import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .functional import forward_spc, backward_spc


class SPC(nn.Module):

    def __init__(self, img_size=32, m=256):
        """
        Initializes the Single Pixel Camera (SPC) model.

        Args:
            img_size (int): Size of the image. Default is 32.
            m (int): Number of measurements. Default is 256.
        """
        super(SPC, self).__init__()

        self.H = nn.Parameter(torch.randn(m, img_size**2))
        self.image_size = img_size

    def forward(self, x, type_calculation="forward"):
        """
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Output tensor after measurement.
        """

        if type_calculation == "forward":
            return forward_spc(x, self.H)
        elif type_calculation == "backward":
            return backward_spc(x, self.H)
        elif type_calculation == "forward_backward":
            return backward_spc(forward_spc(x, self.H), self.H)
        
        else:
            raise ValueError("type_calculation must be 'forward', 'backward' or 'forward_backward'")
        
        
    def ca_reg(self,reg):
        reg_value = reg(self.H)
        return reg_value

    def measurements_reg(self,reg,x):
        y = self(x, type_calculation="forward")
        reg_value = reg(y)
        return reg_value
