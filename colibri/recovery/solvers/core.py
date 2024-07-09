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
        result = self.solve(xtilde, rho)
        return result

    def solve(self, xtilde, rho):
        raise NotImplementedError("Subclasses should implement the solve method.")


class L2L2Solver(Solver):
    r"""

    Base class for linear solvers.


    It describes the close-form solution of the optimization problem.

    .. math::

        \min_{\textbf{x}} \frac{1}{2}||\textbf{y} - \textbf{H}\textbf{x}||_2^2 + \rho||\textbf{x} - \tilde{\textbf{x}}||_2^2

    """

    def __init__(self, y, acquisition_model: BaseOpticsLayer):
        r"""
        Args:
            y (torch.Tensor): Input tensor with shape (B, *)
            acquisition_model (BaseOpticsLayer): Acquisition model
        """

        super(L2L2Solver, self).__init__(y, acquisition_model)

        # vectorized form of y
        y_vec = y.view(y.size(0), -1)

        # batch matrix multiplication
        H = self.acquisition_model.get_sensing_matrix()
        H = H.unsqueeze(0).repeat(y.size(0), 1, 1)

        self.Hty = torch.bmm(H, y_vec.unsqueeze(-1))
        self.HtH = torch.bmm(H, H.permute(0, 2, 1))

    def solve(self, xtilde, rho):
        r"""

        Solves the optimization problem by computing the following expression:

        .. math::            
                    \hat{\textbf{x}} = (\textbf{H}^\top\textbf{H} + \rho \textbf{I})^{-1}(\textbf{H}^\top\textbf{y} + \rho \tilde{\textbf{x}})  

        Args:
            xtilde (torch.Tensor): Regularizaton tensor
            rho (float): Regularization parameter

        Returns:
            torch.Tensor: Recovered tensor
        """

        H_adjoint = self.HtH + rho * torch.eye(self.HtH.shape[1], device=self.HtH.device)
        H_adjoint = torch.inverse(H_adjoint)

        b, c, h, w = xtilde.size()
        x_hat = torch.bmm(H_adjoint, self.Hty + rho * xtilde.view(b, -1, 1)).view(b, c, h, w)

        return x_hat
