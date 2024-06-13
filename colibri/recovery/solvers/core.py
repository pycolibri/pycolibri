import torch 
from colibri.optics import BaseOpticsLayer

class Solver(object):

    r"""
        Base class for all solvers.
    """

    def __init__(self, y, acquisition_model: BaseOpticsLayer):
        r"""
        Initializes the solver.

        Args:
            y (torch.Tensor): Input tensor
            acquisition_model (BaseOpticsLayer): Acquisition model
        """
        pass


    def __call__(self, xtilde, rho):
        pass


class LinearSolver(Solver):

    r"""
        Base class for linear solvers.


        It describes the close-form solution of the optimization problem.

        .. math::

            \min_{x} \frac{1}{2}||y - Hx||_2^2 + \rho||x - xtilde||_2^2

        which has the following solution:

        .. math::
            
                \hat{X} = (H^TH + \rho I)^{-1}(H^Ty + \rho xtilde)    

    """

    def __init__(self, y, acquisition_model: BaseOpticsLayer):
        r"""
        Initializes the linear solver.

        Args:
            y (torch.Tensor): Input tensor
            acquisition_model (BaseOpticsLayer): Acquisition model
        """

        super(LinearSolver, self).__init__(y, acquisition_model)

        # vectorized form of y
        y_vec = y.view(y.size(0), -1)

        # batch matrix multiplication
        H = self.acquisition_model.get_sensing_matrix()
        H = H.unsqueeze(0).repeat(y.size(0), 1, 1)
        
        self.Hty = torch.bmm(H, y_vec.unsqueeze(-1))
        self.HtH = torch.bmm(H, H.permute(0, 2, 1))

    def __call__(self, xtilde, rho):

        Hadj = self.HtH + rho * torch.eye(self.HtH.shape[1], device=self.HtH.device)
        Hadj = torch.inverse(Hadj)

        b, c, h, w = xtilde.size()
        x_hat = torch.bmm(Hadj, self.Hty + rho * xtilde.view(b, -1, 1)).view(b, c, h, w)

        return x_hat