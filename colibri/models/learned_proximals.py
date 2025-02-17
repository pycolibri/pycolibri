""" Learned proximals """

import torch
import torch.nn as nn
from colibri.recovery.terms.prior import Prior

class LearnedPrior(Prior):
    r"""
    Learned prior for hyperspectral image reconstruction.
    
    .. math::
        \boldsymbol{f}^{(k+1)} = S(\boldsymbol{f}^{(k)})

    Args:

        model (nn.Sequential): The neural network acting as the prior.
    
    
        
    """

    def __init__(self, models=None):
        super(LearnedPrior, self).__init__()

        self.count = 0
        self.models = models
        
    def prox(self, x,*args ,**kwargs):

        x = self.models[self.count](x)
        
        self.count += 1

        return x
    def reset(self):
        self.count = 0

class SparseProximalMapping(nn.Module):

    def __init__(self, autoencoder, beta=1e-3):
        '''
        Args:

            autoencoder (nn.Module): The autoencoder model.
            beta (float): The step size for the gradient step. Defaults to 1e-3.

            [FISTA NET paper]
        Returns:
            None
        '''
        super(SparseProximalMapping, self).__init__()

        self.autoencoder = autoencoder
        self.beta = nn.Parameter(beta, requires_grad=True)

    def forward(self,x):
        
        x = self.autoencoder.inc(x)

        for down in self.autoencoder.downs:
            x = down(x)
        
        x = self.autoencoder.bottle(x)

        sp = nn.Softshrink(self.beta)(x)

        for up in self.autoencoder.ups:
            sp = up(sp)
        
        sp = self.autoencoder.outc(sp)
        return sp
    

class OptimizationInspiredLearnedPrior(LearnedPrior):
    r"""
    Optimization-inspired learned prior for hyperspectral image reconstruction.
    
    .. math::
        \boldsymbol{f}^{(k+1)} = \overline{\boldsymbol{\Phi}} \boldsymbol{f}^{(k)} + \epsilon \boldsymbol{f}^{(0)} + \epsilon \eta S(\boldsymbol{f}^{(k)})
    
    Args:
        max_iter (int): Number of iterations (stages).
        model (nn.Module): The neural network acting as the prior.
        prior_args (dict): Arguments to initialize the prior network.
        Phi (torch.Tensor): The measurement matrix.
        epsilon (float): Step size in gradient descent.
        eta (float): Regularization weight for the prior term.
    """
    
    def __init__(self, Phi, max_iter=5, model=None, prior_args=None, epsilon=1e-3, eta=1e-3):
        super(OptimizationInspiredLearnedPrior, self).__init__(max_iter=max_iter, model=model, prior_args=prior_args)
        
        self.Phi = Phi
        self.Phi_T = Phi.transpose(-1, -2)  # Precompute Phi^T
        self.epsilon = epsilon
        self.eta = eta
        self.f0 = None  # Initialization of f^{(0)}
        self.f = None   # Current state of f

    def prox(self, f, g, **kwargs):
        r"""
        Perform the optimization-inspired update based on the learned proximal operator.
        
        Args:
            f (torch.Tensor): Current estimate of hyperspectral image.
            g (torch.Tensor): Observed image (sensor measurements).
        
        Returns:
            torch.Tensor: Updated estimate of the hyperspectral image.
        """
        # Initialization: f^{(0)} = Phi^T g
        if self.count == 0:
            self.f0 = torch.matmul(self.Phi_T, g)
            self.f = self.f0.clone()
        
        # First part: Apply linear transformation \overline{\Phi} to f
        f1 = self.linear_transformation(self.f)
        
        # Second part: Apply the learned prior network S(f)
        f2 = self.model[self.count](self.f)
        
        # Update f based on the unified framework
        self.f = f1 + self.epsilon * self.f0 + self.epsilon * self.eta * f2
        
        self.count += 1
        return self.f

    def reset(self):
        r"""
        Reset the iteration count for multiple optimization runs.
        """
        super(OptimizationInspiredLearnedPrior, self).reset()
        self.f0 = None
        self.f = None
    
    def linear_transformation(self, f):
        r"""
        Compute the linear transformation involving \overline{\boldsymbol{\Phi}}:
        
        .. math::
            \overline{\boldsymbol{\Phi}} = (1 - \epsilon \eta) \boldsymbol{I} - \epsilon \boldsymbol{\Phi}^{\top} \boldsymbol{\Phi}
        
        Args:
            f (torch.Tensor): The current estimate of the hyperspectral image.
        
        Returns:
            torch.Tensor: The result of applying the linear transformation.
        """
        Phi_T_Phi = torch.matmul(self.Phi_T, self.Phi)  # Phi^T * Phi
        
        # Compute \overline{\Phi} f = [(1 - \epsilon * eta) I - \epsilon * Phi^T * Phi] f
        linear_term = (1 - self.epsilon * self.eta) * f - self.epsilon * torch.matmul(Phi_T_Phi, f)
        
        return linear_term



class SparseProximalMapping(nn.Module):
    r"""
    Sparse proximal mapping using an autoencoder network.
    
    This module applies a soft-thresholding function within the autoencoder framework
    to approximate the sparse representation.
    
    Args:
        autoencoder (nn.Module): The autoencoder model.
        beta (float): Step size for the gradient step. Defaults to 1e-3.
    """
    
    def __init__(self, autoencoder, beta=1e-3):
        super(SparseProximalMapping, self).__init__()
        
        self.autoencoder = autoencoder
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=True)

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
        sp = nn.Softshrink(self.beta)(x)

        for up in self.autoencoder.ups:
            sp = up(sp)
        
        sp = self.autoencoder.outc(sp)
        return sp
