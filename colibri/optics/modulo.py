import torch

from colibri.optics.functional import modulo

from .utils import BaseOpticsLayer




class Modulo(BaseOpticsLayer):

    r"""
        Modulo Sensing Operator

        This operator applies the non-linear modulo operation to the input tensor.

        Mathematically, this operator can be described as follows:

        .. math::
                \mathbf{y} = \mathcal{M}_t(\mathbf{x}) = \text{mod}(\mathbf{x}, t)
        
        where :math:`\mathbf{x}\in\xset` is the input tensor, :math:`\mathbf{y}\in\yset` is the output tensor, and :math:`t` is the threshold value.

    """

    def __init__(self, threshold=1.0):
        
        self.threshold = torch.tensor([threshold], requires_grad=False)

        super(Modulo, self).__init__(learnable_optics=threshold, sensing=modulo, backward=None)
    

    def forward(self, x, type_calculation="forward"):
        r"""
        Forward pass of the modulo operator

        Args:
            x (torch.Tensor): Input tensor with shape (B, L, M, N)
            type_calculation (str): Type of calculation to perform. Default: "forward"

        Returns:
            torch.Tensor: Output tensor with shape (B, L, M, N)
        """
        if type_calculation == "backward":
            raise ValueError("The modulo operator does not have a backward pass")

        return super(Modulo, self).forward(x, type_calculation)
        