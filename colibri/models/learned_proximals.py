""" Learned proximals """

import torch.nn as nn
from colibri.recovery.terms.prior import Prior

class LearnedPrior(Prior):

    def __init__(self, max_iter=5, model=None, prior_args=None):
        super(LearnedPrior, self).__init__()

        self.count = 0
        self.model = model
        
    def prox(self, x,*args ,**kwargs):

        x = self.model[self.count](x)
        
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
    

# class SpatialSpectralPrior(nn.Module):
#     '''
#     [DEEP HYPERSPECTRAL SPATIAL SPECTRAL PRIOR]
#     '''
#     pass 




