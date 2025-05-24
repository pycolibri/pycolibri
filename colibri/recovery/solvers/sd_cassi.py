import torch

from colibri.optics import SPC, SD_CASSI
from colibri.optics.functional import forward_spc
from .core import Solver


class L2L2SolverSDCASSI(Solver):
    r"""
        Solver for the SPC acquisition model.

        This module provides a closed-form solution to the following optimization problem:

        .. math::

            \min_{\textbf{X}} \|\textbf{Y} - \textbf{H}\textbf{X}\|_2^2 + \rho\|\textbf{X} - \tilde{\textbf{X}}\|_2^2

        where:
            - :math:`\textbf{X}` is the signal (tensor) to be recovered,
            - :math:`\textbf{Y}` is the observed measurement tensor,
            - :math:`\textbf{H}` is the sensing matrix,
            - :math:`\rho` is the regularization parameter.

        In the context of the SD-CASSI (Spatial-Domain Coded Aperture Snapshot Spectral Imaging) acquisition model,
        :math:`\textbf{X}` is a 3D tensor of shape :math:`(L, M, N)`, where:
            - :math:`L` is the number of spectral channels,
            - :math:`M` and :math:`N` are the spatial dimensions (height and width).

        Here, :math:`\textbf{X}` is vectorized along the spatial dimensions, assuming that the sensing matrix
        :math:`\textbf{H}` is broadcasted across the spectral channels.

        The closed-form solution to the optimization problem is:

        .. math::
            \hat{\textbf{X}} = (\textbf{H}^\top\textbf{H} + \rho \textbf{I})^{-1} \textbf{H}^\top \textbf{Y}

        This implementation follows the fast and efficient computational strategy described in:

        *Fast matrix inversion in compressive spectral imaging based on a tensorial representation*,
        Journal of Electronic Imaging, 33(1), 013034.
        https://doi.org/10.1117/1.JEI.33.1.013034

        Please cite this publication if you use this code in your research.
    """

    def __init__(self, y, acquisition_model: SD_CASSI):
        r"""
        Args:
            y (torch.Tensor): Input tensor with shape (B, K, M, N + L - 1)
            acquisition_model (SD_CASSI): Acquisition model
        """

        super(L2L2SolverSDCASSI, self).__init__(y, acquisition_model)

        self.acquisition_model = acquisition_model
        self.ca = acquisition_model.learnable_optics[0]
        self.Hty = acquisition_model(y, type_calculation="backward")
        self.P = self.computeP(acquisition_model.L)
        self.Q = self.computeQ(acquisition_model.L)

    def solve(self, xtilde, rho):
        X = rho * xtilde + self.Hty
        [Pinv, _] = self.ComputePinv(rho)
        return (1 / rho) * (X - self.acquisition_model(self.IMVM(self.acquisition_model(X), Pinv),
                                                       type_calculation='backward'))

    def computeP(self, L):
        """
        Compute the matrix P that follows the equation (5) from the paper.
        Args:
            L (int): Integer, number of spectral bands.
        Returns:
            torch.Tensor: Matrix P with shape (S, S, M, N + L - 1)
        """
        [S, M, N] = self.ca.shape
        P = torch.zeros(S, S, M, N + L - 1).to(self.ca.device)
        for t in range(S):
            for s in range(t, S):
                # This loop corresponds to the formula: sum_{l=1}^L S^Z_{l-1}(f_t).*S^Z_{l-1}(f_s)
                for l in range(L):
                    P[t, s, :, l:N + l] = P[t, s, :, l:N + l] + self.ca[t] * self.ca[s]

                # Here we use the fact that the matrix is "self-adjoint", so we do not need
                # to compute lower diagonal images
                P[s, t] = P[t, s]

        return P

    def computeQ(self, L):
        """
        Compute the matrix Q that follows the equation (9) from the paper.
        Args:
            L (int): Integer, number of spectral bands.
        Returns:
            torch.Tensor: Matrix Q with shape (L, L, M, N)
        """
        [_, M, N] = self.ca.shape
        Q = torch.zeros(L, L, M, N).to(self.ca.device)
        # we first evaluate the first row of Q's, so k=1 below
        for l in range(L):
            Q[0, l, :, l:N] = torch.sum(self.ca[..., :N - l] * self.ca[..., l:N], dim=0)

        # We now compute remaining indices using the symmetry
        for k in range(L):
            for l in range(k, L):
                # This line uses the Teoplitz structure to get the upper
                # diagonal elements
                Q[k, l] = Q[0, l - k]
                # And this line the formula Q_{l,k}=S_{k-l}[Q_{k,l}],
                # to get the lower diagonal elements
                Q[l, k, :, :N - l + k] = Q[k, l, :, l - k:N]

        return Q

    def IMVM(self, y, P=None):
        """
        Compute the tensor matrix vector multiplication of P and y following the equation (7) from the paper.
        Args:
            y (torch.Tensor): Input tensor with shape (b, S, M, N + L - 1)
            P (torch.Tensor): Tensor P with shape (S, S, M, N + L - 1)
        Returns:
            torch.Tensor: Output tensor with shape (b, S, M, N + L - 1)
        """
        P = self.P if P is None else P
        return torch.sum(P * y[:, None], dim=2)

    def IMVMS(self, x, Q=None):
        """
        Compute the tensor matrix vector multiplication of Q and x following the equation (11) from the paper.
        """
        Q = self.Q if Q is None else Q
        # this is an image - matrix vector multiplication with shifts
        [L, _, M, N] = Q.shape
        y = torch.zeros(x.shape[0], L, M, N).to(Q.device)
        for k in range(L):
            for l in range(k + 1):
                y[:, k, :, :N + l - k] += Q[k, l, :, :N + l - k] * x[:, l, :, k - l:N]

            for l in range(k + 1, L):
                y[:, k, :, l - k:N] += Q[k, l, :, l - k:N] * x[:, l, :, :N - l + k]

        return y

    def ComputePinv(self, rho, P=None):
        """
        Compute the inverse of the tensor (rhoI + P) following the subsection 6.1 from the paper.
        """
        P = self.P if P is None else P
        [S, _, M, NplusLminus1] = P.shape
        e = torch.ones(M, NplusLminus1)
        E = torch.zeros_like(P).to(P.device)

        for t in range(S):
            E[t, t] = e

        R = rho * E + P
        A = rho * E + P
        T = E

        # we first work the lower diagonal rows, just as in standard gaussian elimination
        for t in range(S):
            Rtt = R[t, t].clone()

            # this divides the t'th row with r_tt
            for s in range(S):
                R[t, s] = R[t, s] / Rtt
                T[t, s] = T[t, s] / Rtt

            # this withdraws a multiple of the t'th row from the remaining
            for u in range(t + 1, S):
                # this is the multiple in question
                Rut = R[u, t].clone()
                # and here comes the elimination step
                for s in range(S):
                    R[u, s] = R[u, s] - Rut * R[t, s]
                    T[u, s] = T[u, s] - Rut * T[t, s]

        # this piece does the final "upper triangular" elimination
        for s in reversed(range(1, S)):
            for t in reversed(range(s)):
                for s1 in range(S):
                    T[t, s1] = T[t, s1] - R[t, s] * T[s, s1]

                R[t, s] = R[t, s] - R[t, s] * R[s, s]

        return T, A

    def ComputeQinv(self, rho, Q=None):
        """
        Compute the inverse of the tensor (rhoI + Q) following the subsection 6.2 from the paper.
        """
        Q = self.Q if Q is None else Q
        [L, _, M, N] = Q.shape
        e = torch.ones(M, N)
        E = torch.zeros_like(Q).to(Q.device)

        for t in range(L):
            E[t, t] = e

        R = rho * E + Q
        A = rho * E + Q
        T = E

        for t in range(L):
            Rtt = R[t, t].clone()

            # this divides the tth row with r_tt
            for s in range(L):
                R[t, s] = R[t, s] / Rtt
                T[t, s] = T[t, s] / Rtt

            # this withdraws a suitable multiple of the tth row from the remaining
            for t1 in range(t + 1, L):
                Rt1t = R[t1, t].clone()
                for s in range(L):
                    R[t1, s, :, :N + t - t1] -= Rt1t[:, :N + t - t1] * R[t, s, :, t1 - t: N]
                    T[t1, s, :, :N + t - t1] -= Rt1t[:, :N + t - t1] * T[t, s, :, t1 - t: N]

        # this piece does the final "upper triangular" elimination
        for s in reversed(range(1, L)):
            for t in reversed(range(s)):
                for s1 in range(L):
                    T[t, s1, :, s - t: N] -= R[t, s, :, s - t: N] * T[s, s1, :, : N + t - s]

        return T, A
