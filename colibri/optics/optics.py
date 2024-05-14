from abc import abstractmethod

from torch import nn


class Optics(nn.Module):
    def __init__(self,
                 img_shape,
                 shots,
                 initialization,
                 trainable=False,
                 noise=None,
                 load_ca=None,
                 regularizers=None,
                 constraints=None):
        super(Optics, self).__init__()
        self.img_shape = img_shape
        self.shots = shots
        self.initialization = initialization
        self.trainable = trainable
        self.noise = noise
        self.load_ca = load_ca
        self.regularizers = regularizers
        self.constraints = constraints

        # The following objects must be defined in the child classes

        self.optical_elements = None
        self.forward_operator = None
        self.transpose_operator = None

    def forward(self, x):
        return self.forward_operator(x)

    def transpose(self, y):
        return self.transpose_operator(y)

    def init_forward(self, x):
        return self.transpose_operator(self.forward_operator(x))

    def apply_regularizers(self):
        loss = 0.
        if self.regularizers is not None:
            for regularizer in self.regularizers:
                loss += regularizer(self.optical_elements)

        return loss

    def apply_constraints(self):
        if self.constraints is not None:
            for constraint in self.constraints:
                constraint(self.optical_elements)
