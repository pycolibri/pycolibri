import torch
from colibri_hdsp.optics.functional import forward_color_cassi, backward_color_cassi, forward_dd_cassi, backward_dd_cassi, forward_cassi, backward_cassi


class CASSI(torch.nn.Module):
    """
    Layer that performs the forward and backward operator of coded aperture snapshot spectral imager (CASSI), more information refer to: Compressive Coded Aperture Spectral Imaging: An Introduction: https://doi.org/10.1109/MSP.2013.2278763
    """

    def __init__(self, input_shape, mode = "base", trainable=False, initial_ca=None):
        """
        Args:
            input_shape (tuple): Tuple, shape of the input image (L, M, N).
            mode (str): String, mode of the coded aperture, it can be "base", "dd" or "color"
            trainable (bool): Boolean, if True the coded aperture is trainable
            initial_ca (torch.Tensor): Initial coded aperture with shape (1, M, N, 1)
        """
        super(CASSI, self).__init__()
        self.trainable = trainable
        self.initial_ca = initial_ca

        if mode == "base":
            self.sensing = forward_cassi
            self.backward = backward_cassi
        elif mode == "dd":
            self.sensing = forward_dd_cassi
            self.backward = backward_dd_cassi
        elif mode == "color":
            self.sensing = forward_color_cassi
            self.backward = backward_color_cassi

        self.mode = mode

        self.L, self.M, self.N = input_shape  # Extract spectral image shape

        if self.mode == 'base':
            shape = (1, 1, self.M, self.N)
        elif self.mode == 'dd':
            shape = (1, 1, self.M, self.N + self.L - 1)
        elif self.mode == 'color':
            shape = (1, self.L, self.M, self.N)
        else:
            raise ValueError(f"the mode {self.mode} is not valid")

        if self.initial_ca is None:
            initializer = torch.randn(shape, requires_grad=self.trainable)
        else:
            assert self.initial_ca.shape == shape, f"the start CA shape should be {shape} but is {self.initial_ca.shape}"
            initializer = torch.from_numpy(self.initial_ca).float()

        #Add parameter CA in pytorch manner
        self.ca = torch.nn.Parameter(initializer, requires_grad=self.trainable)

    def forward(self, x, type_calculation="forward"):
        """
        Call method of the layer, it performs the forward or backward operator according to the type_calculation
        Args:
            x (torch.Tensor): Input tensor with shape (1, M, N, L)
            type_calculation (str): String, it can be "forward", "backward" or "forward_backward"
        Returns:
            torch.Tensor: Output tensor with shape (1, M, N + L - 1, 1) if type_calculation is "forward", (1, M, N, L) if type_calculation is "backward, or (1, M, N, L) if type_calculation is "forward_backward
        Raises:
            ValueError: If type_calculation is not "forward", "backward" or "forward_backward"
        """
        if type_calculation == "forward":
            return self.sensing(x, self.ca)

        elif type_calculation == "backward":
            return self.backward(x, self.ca)
        elif type_calculation == "forward_backward":
            return self.backward(self.sensing(x, self.ca), self.ca)

        else:
            raise ValueError("type_calculation must be forward, backward or forward_backward")
        

        
    def ca_reg(self,reg):
        """
        Regularization of the coded aperture.

        Args:
            reg (function): Regularization function.
        
        Returns:
            torch.Tensor: Regularization value.
        """
        reg_value = reg(self.ca)
        return reg_value

    def measurements_reg(self,reg,x):
        """
        Regularization of the measurements.

        Args:
            reg (function): Regularization function.
            x (torch.Tensor): Input image tensor of size (b, c, h, w).

        Returns:
            torch.Tensor: Regularization value.
        """
        y = self.sensing(x, self.ca)
        reg_value = reg(y)
        return reg_value

