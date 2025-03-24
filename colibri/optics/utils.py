import torch


class BaseOpticsLayer(torch.nn.Module):
    r"""
    Base class for CASSI systems.
    """

    def __init__(self, learnable_optics, sensing, backward):
        r"""
        Initializes the BaseOpticsLayer layer.

        Args:
            learnable_optics (torch.Tensor): Coded aperture
            sensing (function): Sensing function.
            backward (function): Backward function.
        """
        super(BaseOpticsLayer, self).__init__()
        self.learnable_optics = learnable_optics
        self.sensing = sensing
        self.backward = backward

    def forward(self, x, type_calculation="forward", **kwargs):
        r"""
        Performs the forward or backward operator according to the type_calculation

        Args:
            x (torch.Tensor): Input tensor
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """

        if type_calculation == "forward":
            return self.sensing(x, self.learnable_optics, **kwargs)

        elif type_calculation == "backward":
            return self.backward(x, self.learnable_optics, **kwargs)
        elif type_calculation == "forward_backward":
            return self.backward(
                self.sensing(x, self.learnable_optics, **kwargs), self.learnable_optics, **kwargs
            )

        else:
            raise ValueError("type_calculation must be forward, backward or forward_backward")

    def weights_reg(self, reg):
        r"""
        Regularization of the coded aperture.

        Args:
            reg (function): Regularization function.

        Returns:
            torch.Tensor: Regularization value.
        """

        reg_value = reg(self.learnable_optics)
        return reg_value

    def output_reg(self, reg, x):
        r"""
        Regularization of the measurements.

        Args:
            reg (function): Regularization function.
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Regularization value.
        """

        y = self.sensing(x, self.learnable_optics)
        reg_value = reg(y)
        return reg_value
