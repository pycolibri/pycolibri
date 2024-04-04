import torch
import torch.nn as nn
from .functional import forward_spc, backward_spc


class SPC(nn.Module):

    r"""
    Single Pixel Camera (SPC).

    
    Mathematically, SPC systems can be described as follows. 

    .. math::

        \mathbf{y} = \forwardLinear_{\learnedOptics}(\mathbf{x}) + \noise

    where :math:`\noise` is the sensor noise, :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal, for SPC, :math:`\xset = \mathbb{R}^{L \times M \times N}` and :math:`\yset = \mathbb{R}^{L \times S}`, and :math:`\forwardLinear_{\learnedOptics}:\xset\rightarrow \yset` is the forward single pixel acquisition and the modulation of the coded aperture, such as

    .. math::
        \begin{align*}
        \forwardLinear_{\learnedOptics}: \mathbf{x} &\mapsto \mathbf{y} \\
                        \mathbf{y}_{s, l} &=  \sum_{i=1}^{M}\sum_{j = 1}^{N} \learnedOptics_{s, i, j} \mathbf{x}_{l, i, j}
        \end{align*}

    with :math:`\learnedOptics \in \{0,1\}^{S \times M \times N}` coded aperture, with :math:`S` the number of measurements and :math:`L` the number of spectral bands.


    
    """
    def __init__(self, input_shape, n_measurements=256, trainable=False, initial_ca=None, **kwargs):
        """
        

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            n_measurements (int): Number of measurements.
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (S, M*N)
        """
        super(SPC, self).__init__()
        _, M, N = input_shape
        self.trainable = trainable
        self.initial_ca = initial_ca
        if self.initial_ca is None:
            initializer = torch.randn((n_measurements, M*N), requires_grad=self.trainable)
        else:
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        self.ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)


    def forward(self, x, type_calculation="forward"):
        """
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (B, L, M, N).

        Returns:
            torch.Tensor: Output tensor after measurement.
        """
        if type_calculation == "forward":
            return forward_spc(x, self.ca)
        elif type_calculation == "backward":
            return backward_spc(x, self.ca)
        elif type_calculation == "forward_backward":
            return backward_spc(forward_spc(x, self.ca), self.ca)
        
        else:
            raise ValueError("type_calculation must be 'forward', 'backward' or 'forward_backward'")
        
        
    def weights_reg(self,reg):
        """
        Regularization of the coded aperture.

        Args:
            reg (function): Regularization function.
        
        Returns:
            torch.Tensor: Regularization value.
        """
        reg_value = reg(self.ca)
        return reg_value

    def output_reg(self,reg,x):
        """
        Regularization of the measurements.

        Args:
            reg (function): Regularization function.
            x (torch.Tensor): Input image tensor of size (B, L, M, N).

        Returns:
            torch.Tensor: Regularization value.
        """
        y = self(x, type_calculation="forward")
        reg_value = reg(y)
        return reg_value
