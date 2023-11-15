import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


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

    def forward(self, x):
        """
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Output tensor after measurement.
        """

        # spatial vectorization
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0, 2, 1)

        # measurement
        H = self.H.unsqueeze(0).repeat(b, 1, 1)
        y = torch.bmm(H, x)
        return y
