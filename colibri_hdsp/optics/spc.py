import torch
from .functional import forward_spc, backward_spc
from .utils import BaseOpticsLayer

class SPC(BaseOpticsLayer):

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
        r"""       

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            n_measurements (int): Number of measurements.
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (S, M*N)
        """
        #super(SPC, self).__init__()
        _, M, N = input_shape
        self.trainable = trainable
        self.initial_ca = initial_ca
        if self.initial_ca is None:
            initializer = torch.randn((n_measurements, M*N), requires_grad=self.trainable)
        else:
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)
        super(SPC, self).__init__(learnable_optics=ca, sensing=forward_spc, backward=backward_spc)


    def forward(self, x, type_calculation="forward"):
        r"""
        Forward propagation through the SPC model.

        Args:
            x (torch.Tensor): Input image tensor of size (B, L, M, N).

        Returns:
            torch.Tensor: Output tensor after measurement of size (B, S, L).
        """
        return super(SPC, self).forward(x, type_calculation)
        