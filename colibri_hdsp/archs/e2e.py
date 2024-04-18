import torch.nn as nn


class E2E(nn.Module):
    def __init__(self, optical_layer: nn.Module, decoder: nn.Module):
        r""" End-to-end model (E2E) for image reconstruction from compressed measurements.
        
        In E2E models, the optical system and the computational decoder are modeled as layers of a neural network, denoted as :math:` \forwardLinear_{\learnedOptics}` and :math:`\mathcal{N}_\theta`, respectively. The optimization problem is formulated as following:

        .. math::
            \begin{equation}
                \begin{aligned}
                    \{ \phi^*, \theta^* \} &= \argmin_{\phi, \theta} \mathcal{L}(\learnedOptics, \theta) \\
                    & \coloneqq \sum_k \mathcal{L}_{\text{task}} ( \mathcal{N}_\theta(\forwardLinear_\learnedOptics(\mathbf{x}_k)), \mathbf{x}_k) + \mu R_1(\learnedOptics) + \lambda R_2(\theta)
                \end{aligned}
            \end{equation},
        
        where the optimal set of parameters of the optical system and the decoder are represented by :math:`\{\learnedOptics^*, \theta^*\}`. The training dataset consists of input-output pairs :math:`\{\mathbf{x}_k, \mathbf{y}_k\}_{k=1}^K`. The loss function :math:`\mathcal{L}_{\text{task}}` is selected based on the task, usually using mean square error (MSE) for reconstruction and cross-entropy for classification. Regularization functions :math:`R_1(\phi)` and :math:`R_2(\theta)`, with parameters :math:`\mu` and :math:`\lambda`, are used to prevent overfitting in the optical system and the decoder.

        For more information refer to: Deep Optical Coding Design in Computational Imaging: A data-driven framework https://doi.org/10.1109/MSP.2022.3200173
        
        Args:
            optical_layer (nn.Module): Optical Layer module.
            decoder (nn.Module): Computational decoder module.
        """
        super(E2E, self).__init__()
        self.optical_layer = optical_layer
        self.decoder = decoder
    
    def forward(self, x):
        r""" Forward pass of the E2E model.

        .. math::
            \begin{equation}
                \begin{aligned}
                    \mathbf{y} &= \forwardLinear_{\learnedOptics}(\mathbf{x}) \\
                    \mathbf{x}_{\text{init}} &= \forwardLinear_{\learnedOptics}^T(\forwardLinear_{\learnedOptics}(\mathbf{y}))\\
                    \hat{\mathbf{x}} &= \mathcal{N}_\theta(\mathbf{x}_{\text{init}})
                \end{aligned}
            \end{equation}
        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N).
        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N).
        """
        y      = self.optical_layer(x) # y = A(x)
        x_init = self.optical_layer(y, type_calculation="backward") # x_init = A^T(y)
        x_hat  = self.decoder(x_init)                  # x_hat = R(x_init)
        return x_hat

