import torch
from torch import nn

# from colibri.recovery.terms.fidelity import L2
from colibri.recovery.fista import Fista
from colibri.models.learned_proximals import LearnedPrior


class UnrollingFISTA(Fista):

    r"""
    Unrolling FISTA Algorithm.
    ===================================================

    The `UnrollingFISTA` class implements the Unrolling FISTA (Fast Iterative Shrinkage-Thresholding Algorithm), which is an optimization algorithm designed to solve inverse problems by iterating over stages in a learned network.

    The optimization problem is formulated as:

    .. math::

        \arg\min_{\theta} \sum_{p=1}^P \left\| \mathcal{N}_{\theta^K} \left( \mathcal{N}_{\theta^{K-1}} \left( \cdots \mathcal{N}_{\theta^1} \left( \forwardLinear_\learnedOptics(\mathbf{x}_p) \right) \right) \right) \right\|_2

    where :math:`\mathcal{N}_{\theta^k}, k = 1, \dots, K` are the stages of the unrolling network. Each stage corresponds to a step in the iterative recovery process of the image or signal.
    """

    def __init__(self, acquistion_model, fidelity=None, max_iters=5, models=None, alpha=1e-3, _lambda=0.1):
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
        self.prior = LearnedPrior(models=models)

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
    

    