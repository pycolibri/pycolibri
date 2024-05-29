import torch

from colibri.optics import SPC
from colibri.optics.functional import forward_spc


class SPCSolver(object):

    def __init__(self, y, acquisition_model: SPC):

        self.y = y
        self.acquisition_model = acquisition_model
        self.Hty = self.acquisition_model(self.y, type_calculation="backward")

    def __call__(self, xtilde, rho):

        H = self.acquisition_model.learnable_optics

        Hadj = torch.matmul(H.permute(1, 0), H) + rho * torch.eye(H.shape[1])
        Hadj = torch.inverse(Hadj)

        b, c, h, w = xtilde.size()
        x_hat = forward_spc( self.Hty + rho * xtilde, Hadj)
        x_hat = x_hat.permute(0, 2, 1)
        x_hat = x_hat.view(b, c, h, w)

        return x_hat