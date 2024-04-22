import torch.nn as nn


class E2E(nn.Module):
    def __init__(self, optical_layer: nn.Module, decoder: nn.Module):
        r""" End-to-end model (E2E) for image reconstruction from compressed measurements.
        
        In E2E models, the optical system and the computational decoder are modeled as layers of a neural network, denoted as :math:`\forwardLinear_{\learnedOptics}` and :math:`\reconnet`, respectively. The optimization problem is formulated as following:

        .. math::
            \begin{equation}
                \begin{aligned}
                    \{ \learnedOptics^*, \theta^* \} &= \arg \min_{\learnedOptics, \theta} \mathcal{L}(\learnedOptics, \theta) \\
                    & \coloneqq \sum_{p=1}^P \mathcal{L}_{\text{task}} (\reconnet(\forwardLinear_\learnedOptics(\mathbf{x}_p)), \mathbf{x}_p) + \lambda \mathcal{R}_1(\learnedOptics) + \mu \mathcal{R}_2(\theta)
                \end{aligned}
            \end{equation}
        
        where the optimal set of parameters of the optical system and the decoder are represented by :math:`\{\learnedOptics^*, \theta^*\}`. The training dataset consists of input-output pairs :math:`\{\mathbf{x}_p, \mathbf{y}_p\}_{p=1}^P`. The loss function :math:`\mathcal{L}_{\text{task}}` is selected based on the task, usually using mean square error (MSE) for reconstruction and cross-entropy for classification. Regularization functions :math:`\mathcal{R}_1(\learnedOptics)` and :math:`\mathcal{R}_2(\theta)`, with parameters :math:`\lambda` and :math:`\mu`, are used to prevent overfitting in the optical system and the decoder.

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
                    \mathbf{x}_{\text{init}} &= \forwardLinear_{\learnedOptics}^\top(\forwardLinear_{\learnedOptics}(\mathbf{y}))\\
                    \hat{\mathbf{x}} &=\reconnet(\mathbf{x}_{\text{init}})
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

