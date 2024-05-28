import torch
from torch import nn


class PnP(nn.Module):
    r"""
    Plug-and-Play (PnP) algorithm for solving the optimization problem

    .. math::
        \begin{equation}
            \underset{\mathbf{x}}{\text{min}} \quad \frac{1}{2}||\mathbf{y} - \forwardLinear (\mathbf{x})||^2 + \lambda||\mathbf{x}||_1
        \end{equation}
    """

    def __init__(self, fidelity, prior, aquisition_model, algo_params, solver="inverse"):

        super(PnP, self).__init__()

        self.fidelity         = fidelity
        self.aquisition_model = aquisition_model
        self.prior            = prior
        self.algo_params      = algo_params
        self.solver           = solver


    def forward(self, y, x0=None, verbose=False):

        InverseProblem = lambda y, algo_params: None
        GradientDescent = lambda y, aquisition_model, fidelity, algo_params: None

        # Initialize the solution
        if self.solver == "inverse":
            x_solver = InverseProblem(y, self.aquisition_model,  self.algo_params)
        else:
            x_solver = GradientDescent(y, self.aquisition_model, self.fidelity, self.algo_params)

        if x0 is None:
            x0 = torch.zeros_like(y)
        
        
        u_t = torch.zeros_like(x0)
        v_t = x0

        for i in range(self.algo_params["max_iters"]):

            # x-subproblem update
            xtilde = v_t - u_t
            x_t    = x_solver.update(xtilde, x_t)

            # v-subproblem update
            vtilde = x_t + u_t
            v_t    = self.prior.prox(vtilde, self.algo_params["lambda"])

            # u-subproblem update
            u_t = u_t + x_t - v_t

            if verbose:
                error = self.fidelity.forward(x_t, y, self.H).item()
                print("Iter: ", i, "fidelity: ", error)

        x_hat = v_t

        return x_hat


