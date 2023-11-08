
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_regularizer(regularizer, layer):
    """
    Compute a regularization term for a given layer's weight tensor.

    Args:
        regularizer (nn.Module): A PyTorch regularization module.
        layer (nn.Module): A PyTorch layer for which to compute the regularization term.

    Returns:
        torch.Tensor: Regularization term calculated using the provided regularizer.
    """
    x = layer.weight
    return regularizer(x)

class Reg_Binary(nn.Module):
    """
    Binary Regularization for Neural Network Weights.

    Args:
        parameter (float): Regularization parameter.
        min_v (float): Minimum value for weight clipping.
        max_v (float): Maximum value for weight clipping.
    """

    def __init__(self, parameter=10, min_v=0, max_v=1):
        super(Reg_Binary, self).__init__()
        self.parameter = parameter
        self.min_v = min_v
        self.max_v = max_v

    def forward(self, x):
        """
        Compute binary regularization term.

        Args:
            x (torch.Tensor): Input tensor (layer's weight).

        Returns:
            torch.Tensor: Binary regularization term.
        """
        regularization = self.parameter * (torch.sum(torch.mul(torch.square(x - self.min_v), torch.square(x - self.max_v))))
        return regularization

class Reg_Transmittance(nn.Module):
    """
    Transmittance Regularization for Neural Network Weights.

    Args:
        parameter (float): Regularization parameter.
        t (float): Target transmittance value.
    """

    def __init__(self, parameter=10, t=0.8):
        super(Reg_Transmittance, self).__init__()
        self.parameter = parameter
        self.t = t

    def forward(self, x):
        """
        Compute transmittance regularization term.

        Args:
            x (torch.Tensor): Input tensor (layer's weight).

        Returns:
            torch.Tensor: Transmittance regularization term.
        """
        transmittance = self.parameter * (torch.square(torch.div(torch.sum(x), torch.numel(x)) - self.t))
        return transmittance

class Correlation(nn.Module):
    """
    Correlation Regularization for Neural Network Layers.

    Args:
        batch_size (int): Batch size used for reshaping.
        param (float): Regularization parameter.
    """

    def __init__(self, batch_size=128, param=0.001):
        super(Correlation, self).__init__()
        self.batch_size = batch_size
        self.param = param

    def forward(self, inputs):
        """
        Compute correlation regularization term.

        Args:
            inputs (tuple): Tuple containing two input tensors (x and y).

        Returns:
            torch.Tensor: Correlation regularization term.
        """
        x, y = inputs
        x_reshaped = x.view(self.batch_size, -1)
        y_reshaped = y.view(self.batch_size, -1)

        Cxx = torch.mm(x_reshaped, x_reshaped.t()) / self.batch_size
        Cyy = torch.mm(y_reshaped, y_reshaped.t()) / self.batch_size

        loss = torch.norm(Cxx - Cyy, p=2)
        return loss

class KLGaussian(nn.Module):
    """
    KL Divergence Regularization for Gaussian Distributions.

    Args:
        mean (float): Target mean for the Gaussian distribution.
        stddev (float): Target standard deviation for the Gaussian distribution.
    """

    def __init__(self, mean=1e-2, stddev=2.0):
        super(KLGaussian, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, y):
        """
        Compute KL divergence regularization term.

        Args:
            y (torch.Tensor): Input tensor representing a Gaussian distribution.

        Returns:
            torch.Tensor: KL divergence regularization term.
        """
        z_mean = torch.mean(y, 0)
        z_log_var = torch.log(torch.std(y, 0))
        kl_loss = -0.5 * torch.mean(
            z_log_var - torch.log(self.stddev) - (torch.exp(z_log_var) + torch.pow(z_mean - self.mean, 2)) / (
                    self.stddev ** 2) + 1)
        return kl_loss

class MinVariance(nn.Module):
    """
    Minimum Variance Regularization for Neural Network Layers.

    Args:
        param (float): Regularization parameter.
    """

    def __init__(self, param=1e-2):
        super(MinVariance, self).__init__()
        self.param = param

    def forward(self, y):
        """
        Compute minimum variance regularization term.

        Args:
            y (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Minimum variance regularization term.
        """
        std_dev = torch.std(y, dim=0)
        var_loss = torch.norm(std_dev, 2)
        return var_loss




    
