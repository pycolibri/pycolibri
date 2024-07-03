import torch
from torch import nn

from .solvers import SOLVERS, get_solver

from colibri.recovery.terms.fidelity import L2
from colibri.recovery.terms.prior import Sparsity

class PnP_ADMM(nn.Module):
    r"""
    Plug-and-Play (PnP) algorithm with Alternating Direction Method of Multipliers (ADMM) formulation.

    The PnP algorithm solves the optimization problem:

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    Implementation based on the formulation of authors in https://doi.org/10.1109/TCI.2016.2629286
    """

    def __init__(self, acquisition_model, fidelity=L2(), prior=Sparsity("dct"), solver="close", max_iters=20, _lambda=0.1, rho=0.1, alpha=0.01):
        r"""
        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            max_iters (int): The maximum number of iterations for the FISTA algorithm. Defaults to 20.
            _lambda (float): The regularization parameter for the prior term. Defaults to 0.1.
            rho (float): The penalty parameter for the ADMM formulation. Defaults to 0.1.
            alpha (float): The step size for the gradient step. Defaults to 1e-3.
            

        Returns:
            None
        """


        super(PnP_ADMM, self).__init__()

        self.fidelity         = fidelity
        self.acquisition_model = acquisition_model
        self.prior            = prior
        self.solver           = solver

        self.max_iters        = max_iters
        self._lambda          = _lambda
        self.rho              = rho
        self.alpha            = alpha

        self.H = lambda x: self.acquisition_model.forward(x)


    def forward(self, y, x0=None, verbose=False):
        r"""Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        # Initialize the solution
        if self.solver == "close":
            ClosedSolution = get_solver(self.acquisition_model)
            x_solver = ClosedSolution(y, self.acquisition_model)

        else:
            GradientDescent = lambda x, xt: x - self.alpha * self.fidelity.grad(x, y, self.H)  - self.rho * (x - xt)
            x_solver = GradientDescent

        if x0 is None:
            x0 = torch.zeros_like(y)
        
        u_t = torch.zeros_like(x0)
        v_t = x0

        for i in range(self.max_iters):

            # x-subproblem update
            xtilde = v_t - u_t
            x_t    = x_solver(xtilde, self.rho) if self.solver == "close" else x_solver(x_t, xtilde)

            # v-subproblem update
            vtilde = x_t + u_t
            v_t    = self.prior.prox(vtilde, self._lambda)

            # u-subproblem update
            u_t = u_t + x_t - v_t

            if verbose:
                error = self.fidelity.forward(x_t, y, self.H).item()
                print("Iter: ", i, "fidelity: ", error)

        return v_t


