import torch.nn as nn


class E2E(nn.Module):
    """
        End-to-end model for image reconstruction from compressed measurements.
    """
    def __init__(self, sensing: nn.Module, reconstruction: nn.Module):
        """ End-to-end model for image reconstruction from compressed measurements.
        Args:
            sensing (nn.Module): Sensing module.
            reconstruction (nn.Module): Reconstruction module.
        """


        super(E2E, self).__init__()
        self.sensing = sensing
        self.reconstruction = reconstruction
    
    def forward(self, x):

        y      = self.sensing(x) # y = A(x)
        x_init = self.sensing(y, type_calculation="backward") # x_init = A^T(y)
        x_hat  = self.reconstruction(x_init)                  # x_hat = R(x_init)
        return x_hat

