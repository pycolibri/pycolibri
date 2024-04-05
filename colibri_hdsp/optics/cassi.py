import torch
from colibri_hdsp.optics.functional import forward_color_cassi, backward_color_cassi, forward_dd_cassi, backward_dd_cassi, forward_sd_cassi, backward_sd_cassi
from .utils import BaseOpticsLayer

    
class SD_CASSI(BaseOpticsLayer):
    r"""
    Single Disperser Coded Aperture Snapshot Spectral Imager (SD-CASSI)

    CASSI systems allow for the capture of spatio-spectral information through spatial coding of light and spectral dispersion through a prism. 
    
    Mathematically, CASSI systems can be described as follows. 

    .. math::

        \mathbf{y} = \forwardLinear_{\learnedOptics}(\mathbf{x}) + \noise

    where :math:`\noise` is the sensor noise, :math:`\mathbf{x}\in\xset` is the input optical field, :math:`\mathbf{y}\in\yset` are the acquired signal, for CASSI, :math:`\xset = \mathbb{R}^{L \times M \times N}` and :math:`\yset = \mathbb{R}^{M \times N+L-1}`, and :math:`\forwardLinear_{\learnedOptics}:\xset\rightarrow \yset` is the forward operator of the prism dispersion and the modulation of the coded aperture, such as

    .. math::
        \begin{align*}
        \forwardLinear_{\learnedOptics}: \mathbf{x} &\mapsto \mathbf{y} \\
                        \mathbf{y}_{i, j+l-1} &=  \sum_{l=1}^{L} \learnedOptics_{i, j} \mathbf{x}_{i, j, l}
        \end{align*}

    with :math:`\learnedOptics \in \{0,1\}^{M \times N}` coded aperture,


    """

    def __init__(self, input_shape, trainable=False, initial_ca=None, **kwargs):
        r"""
        Initializes the SD_CASSI layer.

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, 1, M, N)
        """
        
        self.trainable = trainable
        self.initial_ca = initial_ca


        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        shape = (1, 1, self.M, self.N)
        if self.initial_ca is None:
            initializer = torch.randn(shape, requires_grad=self.trainable) 
        else:
            assert self.initial_ca.shape == shape, f"the start CA shape should be {shape} but is {self.initial_ca.shape}"
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)

        super(SD_CASSI, self).__init__(learnable_optics=ca, sensing=forward_sd_cassi, backward=backward_sd_cassi)

    def forward(self, x, type_calculation="forward"):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (1, 1, M, N + L - 1) if type_calculation is "forward", (1, L, M, N) if type_calculation is "backward, or "forward_backward
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(SD_CASSI, self).forward(x, type_calculation)

class DD_CASSI(BaseOpticsLayer):
    r"""
    
    Coming soon
    """

    def __init__(self, input_shape, trainable=False, initial_ca=None, **kwargs):
        r"""
        Initializes the DD_CASSI layer.

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, 1, M, N + L - 1)
        """
        self.trainable = trainable
        self.initial_ca = initial_ca


        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        shape = (1, 1, self.M, self.N + self.L - 1)

        if self.initial_ca is None:
            initializer = torch.randn(shape, requires_grad=self.trainable)
        else:
            assert self.initial_ca.shape == shape, f"the start CA shape should be {shape} but is {self.initial_ca.shape}"
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)
        super(DD_CASSI, self).__init__(learnable_optics=ca, sensing=forward_dd_cassi, backward=backward_dd_cassi)


    def forward(self, x, type_calculation="forward"):
        r"""
        Call method of the layer, it performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (1, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (1, 1, M, N + L - 1) if type_calculation is "forward", (1, L, M, N) if type_calculation is "backward, or "forward_backward
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(DD_CASSI, self).forward(x, type_calculation)


class C_CASSI(BaseOpticsLayer):
    r"""
    Coming soon

    """

    def __init__(self, input_shape, trainable=False, initial_ca=None, **kwargs):
        r"""
        Initializes the C_CASSI layer.

        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, L, M, N)
        """
        self.trainable = trainable
        self.initial_ca = initial_ca


        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        shape = (1, self.L, self.M, self.N)

        if self.initial_ca is None:
            initializer = torch.randn(shape, requires_grad=self.trainable)
        else:
            assert self.initial_ca.shape == shape, f"the start CA shape should be {shape} but is {self.initial_ca.shape}"
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)
        super(C_CASSI, self).__init__(learnable_optics=ca, sensing=forward_color_cassi, backward=backward_color_cassi)

    def forward(self, x, type_calculation="forward"):
        r"""
        Call method of the layer, it performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Measurement with shape (B, 1, M, N + L - 1)  if type_calculation is "forward", (1, L, M, N) if type_calculation is "backward, or "forward_backward
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        return super(C_CASSI, self).forward(x, type_calculation)

        
