import torch
from torch import nn

from colibri.recovery.terms.fidelity import L2


class UnrollingFISTA(nn.Module):

    '''
    FISTA Unrolling algorithm
    '''

    def __init__(self, acquistion_model, fidelity=L2(), max_iters=5, prior=None, alpha=1e-3, rho=1.0):
        '''
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the unrolling algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.

        Returns:
            None
        '''
        super(UnrollingFISTA, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model

        self.H = lambda x: self.acquistion_model.forward(x)

        self.max_iters = max_iters
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(alpha),requires_grad=True)] * max_iters)
        self.rho = nn.ParameterList([nn.Parameter(torch.tensor(rho),requires_grad=True)] * max_iters)
        self.prior = nn.ModuleList([prior] * max_iters) 

    def forward(self, y, x0=None):
        '''
        Runs the unrolling algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        '''

        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        z = x.clone()

        for i in range(self.max_iters):
            x_old = x.clone()

            # gradient step
            x = z - self.alpha[i] * self.fidelity.grad(z, y, self.H) 

            # deep proximal step
            x = self.prior[i](x)

            # update z
            z = x + self.rho[i] * (x - x_old)

        return x
    

    

if __name__ == '__main__':
    from colibri.models.autoencoder import Autoencoder
    # from colibri.models.learned_proximals import SparseProximalMapping
    from colibri.recovery.terms.fidelity import L2
    from colibri.optics.spc import SPC

    H = SPC([1,256,256],10)

    autoencoder = Autoencoder(1,1)
    # sp = SparseProximalMapping(autoencoder)

    unrolling = UnrollingFISTA(alpha=0.01, rho=1.0, max_iters=7, fidelity=L2(), prior=autoencoder, acquistion_model=H)

    xgt = torch.randn(1,1,256,256)

    y = H(xgt,'forward')    

    x0 = H(y,'backward')
    x = unrolling(y, x0)
    print('Initial guess: ',  torch.norm(x0 - xgt).item())
    print('Reconstructed: ', torch.norm(x - xgt).item())
    # print(x.shape)


