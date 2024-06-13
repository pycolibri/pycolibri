import torch

from colibri.optics import SPC
from colibri.optics.functional import forward_spc
from .core import Solver


class SPCSolver(Solver):

    def __init__(self, y, acquisition_model: SPC):

        super(SPCSolver, self).__init__(y, acquisition_model)


        self.Hty = acquisition_model(y, type_calculation="backward")
        self.H   = acquisition_model.learnable_optics

    def __call__(self, xtilde, rho):
        Hadj = torch.matmul(self.H.permute(1, 0), self.H) + rho * torch.eye(self.H.shape[1])
        Hadj = torch.inverse(Hadj)

        b, c, h, w = xtilde.size()
        x_hat = forward_spc( self.Hty + rho * xtilde, Hadj)
        x_hat = x_hat.permute(0, 2, 1)
        x_hat = x_hat.view(b, c, h, w)

        return x_hat