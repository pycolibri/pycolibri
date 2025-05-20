""" Learned proximals """

import torch
import torch.nn as nn
from colibri.recovery.terms.prior import LearnedPrior
from colibri.models.autoencoder import Autoencoder
def softshrink(x, lambd):
    mask1 = x > lambd
    mask2 = x < -lambd
    out = torch.zeros_like(x)
    out += mask1.float() * -lambd + mask1.float() * x
    out += mask2.float() * lambd + mask2.float() * x
    return out


class SparseProximalMapping(nn.Module):
    r"""
    Sparse proximal mapping using an autoencoder network.
    
    This module applies a soft-thresholding function within the autoencoder framework
    to approximate the sparse representation.
    
    .. math::
    `\mathbf{z} = \mathcal{D}(\mathrm{soft}(\mathcal{E}(\mathbf{f}),\beta))`
    
    Args:
        autoencoder (nn.Module): The autoencoder model.
        beta (float): Step size for the gradient step. Defaults to 1e-3.
    """
    
    def __init__(self, autoencoder_args, beta=1e-3):
        super(SparseProximalMapping, self).__init__()
        
        self.autoencoder = Autoencoder(**autoencoder_args)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)
        self.sthresh = nn.Softshrink(self.beta)

    def forward(self, x):
        r"""
        Forward pass of the autoencoder with soft-thresholding applied.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after sparse proximal mapping.
        """
        x = self.autoencoder.inc(x)

        for down in self.autoencoder.downs:
            x = down(x)
        
        x = self.autoencoder.bottle(x)
        
        # Apply soft-shrinkage with learned beta
        sp = softshrink(x,self.beta)

        for up in self.autoencoder.ups:
            sp = up(sp)
        
        sp = self.autoencoder.outc(sp)
        return sp
