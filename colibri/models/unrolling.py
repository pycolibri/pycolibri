import torch
from torch import nn

# from colibri.recovery.terms.fidelity import L2
from colibri.recovery.fista import Fista
from . import LearnedPrior
class UnrollingFISTA(Fista):

    '''
    FISTA Unrolling algorithm
    '''

    def __init__(self, acquistion_model, fidelity=None, max_iters=5, model=None, alpha=1e-3,  prior_args=None, _lambda=0.1):
        '''
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the unrolling algorithm. Defaults to 5.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.

        Returns:
            None
        '''
        super(UnrollingFISTA, self).__init__(acquistion_model=acquistion_model,fidelity=fidelity, max_iters=max_iters, alpha=alpha, _lambda=_lambda)

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model

        self.H = lambda x: self.acquistion_model.forward(x)

        self.max_iters = max_iters
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(alpha),requires_grad=True)] * max_iters)
        self.t_squence = nn.ParameterList([nn.Parameter(torch.tensor(1.0),requires_grad=True)] * max_iters)
        self.prior = LearnedPrior(max_iter=max_iters, model=model, prior_args=prior_args)

    def forward(self, y, x0=None):
        '''
        Runs the unrolling algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The measurement data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        '''

        return super(UnrollingFISTA, self).forward(y, x0=x0)
    

    

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

