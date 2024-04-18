import torch.nn as nn


class E2E(nn.Module):
    def __init__(self, optical_layer: nn.Module, decoder: nn.Module):
        r""" End-to-end model (E2E) for image reconstruction from compressed measurements.
        
        In E2E models, the optical system and the computational decoder are modeled as layers of a neural network, denoted as :math:`\mathcal{M}_\phi` and :math:`\mathcal{N}_\theta`, respectively. The optimization problem is formulated as following:

        .. math::
            \begin{equation}
                \begin{aligned}
                    \{ \phi^*, \theta^* \} &= \argmin_{\phi, \theta} \mathcal{L}(\phi, \theta) \\
                    & \coloneqq \sum_k \mathcal{L}_{\text{task}} ( \mathcal{N}_\theta(\mathcal{M}_\phi(\mathbf{x}_k)), \mathbf{x}_k) + \rho R_\rho(\phi) + \sigma R_\sigma(\theta)
                \end{aligned}
            \end{equation},
        
        where the optimal set of parameters of the optical system and the decoder are represented by :math:`\{\phi^*, \theta^*\}`. The training dataset consists of input-output pairs :math:`\{\mathbf{x}_k, \mathbf{y}_k\}_{k=1}^K`. The loss function :math:`\mathcal{L}_{\text{task}}` is selected based on the task, usually using mean square error (MSE) for reconstruction and cross-entropy for classification. Regularization functions :math:`R_\rho(\phi)` and :math:`R_\sigma(\theta)`, with parameters :math:`\rho` and :math:`\sigma`, are used to prevent overfitting in the optical system and the decoder.

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

