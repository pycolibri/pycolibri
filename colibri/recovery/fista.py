import torch
from torch import nn


class Fista(nn.Module):
    r"""
    FISTA algorithm for solving the optimization problem

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}

    where :math:`\forwardLinear` is the forward model, :math:`\mathbf{y}` is the data to be reconstructed, :math:`\lambda` is the regularization parameter and :math:`||\cdot||_1` is the L1 norm.

    The FISTA algorithm is an iterative algorithm that solves the optimization problem by performing a gradient step and a proximal step.

    .. math::
        \begin{align*}
         \mathbf{x}_{k+1} &= \text{prox}_{\lambda||\cdot||_1}( \mathbf{z}_k - \alpha \nabla f( \mathbf{z}_k)) \\
        t_{k+1} &= \frac{1 + (1 + 4t_k^2)^{0.5}}{2} \\
        \mathbf{z}_{k+1} &=  \mathbf{x}_{k+1} + \frac{t_k-1}{t_{k+1}}( \mathbf{x}_{k} - \mathbf{x}_{k-1})
        \end{align*}

    where :math:`\alpha` is the step size and :math:`f` is the fidelity term.

    """

    def __init__(self, fidelity, prior, acquistion_model, algo_params, transform):
        """Initializes the Fista class.

        Args:

            fidelity (nn.Module): The fidelity term in the optimization problem. This is a function that measures the discrepancy between the data and the model prediction.
            prior (nn.Module): The prior term in the optimization problem. This is a function that encodes prior knowledge about the solution.
            acquistion_model (nn.Module): The acquisition model of the imaging system. This is a function that models the process of data acquisition in the imaging system.
            algo_params (dict): A dictionary containing the parameters for the optimization algorithm. For example, it could contain the tolerance for the stopping criterion.
            transform (object): The transform to be applied to the image. This is a function that transforms the image into a different domain, for example, the DCT domain.

        Returns:
            None
        """
        super(Fista, self).__init__()

        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior
        self.algo_params = algo_params
        self.transform = transform

        self.H = lambda alpha: self.acquistion_model.forward(self.transform.inverse(alpha))
        self.tol = algo_params["tol"]

    def forward(self, y, x0=None, verbose=False):
        """Runs the FISTA algorithm to solve the optimization problem.

        Args:
            y (torch.Tensor): The data to be reconstructed.
            x0 (torch.Tensor, optional): The initial guess for the solution. Defaults to None.

        Returns:
            torch.Tensor: The reconstructed image.
        """

        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        t = 1
        z = x.clone()

        for i in range(self.algo_params["max_iter"]):
            x_old = x.clone()
            z_old = z.clone()

            # gradient step
            x = z - self.algo_params["alpha"] * self.fidelity.grad(z, y, self.H)

            # proximal step
            x = self.prior.prox(x, self.algo_params["lambda"])

            # FISTA step
            t_old = t
            t = (1 + (1 + 4 * t_old**2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

            error = self.fidelity.forward(x, y, self.H).item()
            
            if verbose:
                print("Iter: ", i, "fidelity: ", error)

        x_hat = self.transform.inverse(x)
        return x_hat
