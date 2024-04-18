import torch.nn as nn


class E2E(nn.Module):
    def __init__(self, optical_layer: nn.Module, decoder: nn.Module):
        """ End-to-end model for image reconstruction from compressed measurements.
        For more information refer to: Deep Optical Coding Design in Computational Imaging: A data-driven framework https://doi.org/10.1109/MSP.2022.3200173
        
        Args:
            optical_layer (nn.Module): Optical Layer module.
            decoder (nn.Module): Computational decoder module.
        """
        super(E2E, self).__init__()
        self.optical_layer = optical_layer
        self.decoder = decoder
    
    def forward(self, x):

        y      = self.optical_layer(x) # y = A(x)
        x_init = self.optical_layer(y, type_calculation="backward") # x_init = A^T(y)
        x_hat  = self.decoder(x_init)                  # x_hat = R(x_init)
        return x_hat

