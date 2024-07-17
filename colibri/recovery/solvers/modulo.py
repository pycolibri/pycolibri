import torch

from colibri.optics import Modulo
from colibri.optics.functional import modulo
from colibri.optics.utils import BaseOpticsLayer
from .core import Solver

import torch.nn.functional as F
import torch_dct as dct


center_modulo = lambda x, t: modulo(x  + t/2, t) - t/2

class L2L2SolverModulo(Solver):
    r"""
        Solver for the Modulo acquisition model.
    """


    def __init__(self, y, acquisition_model: Modulo):
        
        threshold = acquisition_model.threshold

        Mdx_y = F.pad( center_modulo(torch.diff(y, 1, dim=-1), threshold), (1, 0), mode='constant')
        Mdy_y = F.pad( center_modulo(torch.diff(y, 1, dim=-2), threshold), (0, 0, 1, 0), mode='constant')

        DTMDy =  - ( torch.diff(F.pad(Mdx_y, (0, 1)), 1, dim=-1) + torch.diff(F.pad(Mdy_y, (0,0, 0, 1)), 1, dim=-2) )

        self.DTMDy = DTMDy
        self.threshold = threshold

    def solve(self, xtilde, rho, normalize=True):

        if xtilde is None:
            rho = 0.0
            xtilde = torch.zeros_like(self.DTMDy)
        
        psi = self.DTMDy + (rho / 2) * xtilde
        dct_psi = dct.dct_2d(psi, norm='ortho')

        NX, MX = dct_psi.shape[-1], dct_psi.shape[-2]
        I, J = torch.meshgrid(torch.arange(0, MX), torch.arange(0, NX), indexing="ij")
        I, J = I.to(dct_psi.device), J.to(dct_psi.device)

        I, J = I.unsqueeze(0).unsqueeze(0), J.unsqueeze(0).unsqueeze(0)

        denom = 2 * ( ( rho / 4 ) + 2 - ( torch.cos(torch.pi * I / MX ) + torch.cos(torch.pi * J / NX ) ) )
        denom = denom.to(dct_psi.device)

        dct_phi = dct_psi / denom
        dct_phi[..., 0, 0] = 0

        phi = dct.idct_2d(dct_phi, norm='ortho')

        if normalize:
            phi = phi - phi.min()
            phi = phi / phi.max()

        return phi
