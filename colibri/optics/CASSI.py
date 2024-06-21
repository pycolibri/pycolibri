import torch

import colibri.optics.functional as F
from colibri.optics.optics import Optics

CA_SHAPE = {
    'sd_cassi': lambda M, N, L: (M, N),
    'dd_cassi': lambda M, N, L: (M, N + L - 1),
    'color_cassi': lambda M, N, L: (M, N, L)
}

CA_INITIALIZATION = {
    'random': torch.randn,
    'uniform': torch.rand,
    'ones': torch.ones,
    'zeros': torch.zeros
}


def initialize_ca(name, img_shape, trainable=False, initialization=None, load_ca=None):
    assert len(img_shape) == 3, "The image shape must be (M, N, L)"

    ca_shape = CA_SHAPE[name](img_shape)
    if load_ca is not None:
        ca = torch.load(load_ca)
        assert ca.shape == ca_shape(img_shape), f"The coded aperture must have the shape {ca_shape}"

    else:
        ca_init = CA_INITIALIZATION[initialization]
        ca = ca_init(*ca_shape)

    return torch.nn.Parameter(ca, requires_grad=trainable)


class SDCASSI(Optics):
    def __init__(self,
                 img_shape,
                 shots,
                 initialization,
                 trainable=False,
                 noise=None,
                 load_ca=None,
                 regularizers=None,
                 constraints=None):
        super(SDCASSI, self).__init__(img_shape, shots, initialization, trainable, noise, load_ca, regularizers,
                                      constraints)

        ca = initialize_ca('sd_cassi', img_shape, trainable=trainable, initialization=initialization,
                           load_ca=load_ca)

        self.optical_elements = {'ca': ca}
        self.forward_operator = lambda x: F.forward_sd_cassi(x, ca)
        self.transpose_operator = lambda x: F.backward_sd_cassi(x, ca)

    def forward(self, x):
        super().forward(x)
