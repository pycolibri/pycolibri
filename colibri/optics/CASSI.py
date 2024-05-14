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

CASSI_OPERATORS = {
    'sd_cassi': {'forward': F.forward_sd_cassi, 'transpose': F.backward_sd_cassi},
    'dd_cassi': {'forward': F.forward_dd_cassi, 'transpose': F.backward_dd_cassi},
    'color_cassi': {'forward': F.forward_color_cassi, 'transpose': F.backward_color_cassi}
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


class CASSI(Optics):
    def __init__(self,
                 name,
                 img_shape,
                 shots,
                 initialization,
                 trainable=False,
                 noise=None,
                 load_ca=None,
                 regularizers=None,
                 constraints=None):
        super(CASSI, self).__init__(img_shape, shots, initialization, trainable, noise, load_ca, regularizers,
                                    constraints)

        ca = initialize_ca(name, img_shape, trainable=trainable, initialization=initialization, load_ca=load_ca)
        self.optical_elements = {'ca': ca}

        cassi_operator = CASSI_OPERATORS[name]
        self.forward_operator = lambda x: cassi_operator['forward'](x, self.optical_elements['ca'])
        self.transpose_operator = lambda x: cassi_operator['backward'](x, self.optical_elements['ca'])
