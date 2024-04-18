
import torch
import torch.nn as nn
import torch.nn.functional as F



class Reg_Binary(nn.Module):
    r"""
    Binary Regularization for Coded Aperture Design.

    Code adapted from  Bacca, Jorge, Tatiana Gelvez-Barrera, and Henry Arguello. 
    "Deep coded aperture design: An end-to-end approach for computational imaging tasks." 
    IEEE Transactions on Computational Imaging 7 (2021): 1148-1160.
    The regularizer computes:

    .. math::

        \begin{equation*}
        R(x) = \mu\sum_{i=1}^{n} (x_i - \text{min_v})^2(x_i - \text{max_v})^2  
        \end{equation*}

    where :math:`\mu` is the regularization parameter and :math:`\text{min_v}` and :math:`\text{max_v}` are the minimum and maximum values for the weights, respectively.
    """


    def __init__(self, parameter=10, min_v=0, max_v=1):
        """

        Args:
            parameter (float): Regularization parameter.
            min_v (float): Minimum value for weight clipping.
            max_v (float): Maximum value for weight clipping.
        """
        
        super(Reg_Binary, self).__init__()
        self.parameter = parameter
        self.min_v = min_v
        self.max_v = max_v
        self.type_reg = 'ca'

    def forward(self, x):
        r"""
        Compute binary regularization term.
        
        Args:
            x (torch.Tensor): Input tensor (layer's weight).

        Returns:
            torch.Tensor: Binary regularization term.
        """
        regularization = self.parameter * (torch.sum(torch.mul(torch.square(x - self.min_v), torch.square(x - self.max_v))))
        return regularization

class Reg_Transmittance(nn.Module):
    r"""
    Transmittance Regularization for Coded Apeuture Design.
    

    Code adapted from  Bacca, Jorge, Tatiana Gelvez-Barrera, and Henry Arguello. 
    "Deep coded aperture design: An end-to-end approach for computational imaging tasks." 
    IEEE Transactions on Computational Imaging 7 (2021): 1148-1160.
    
    The regularizer computes:

    .. math::
        \begin{equation*}
        R(x) = \mu \left(\sum_{i=1}^{n}\frac{x_i}{n}-t\right)^2 
        \end{equation*}

    where :math:`\mu` is the regularization parameter and :math:`t` is the target transmittance value.
    """

    def __init__(self, parameter=10, t=0.8):
        """
        Args:
            parameter (float): Regularization parameter.
            t (float): Target transmittance value.
        """

        super(Reg_Transmittance, self).__init__()
        self.parameter = parameter
        self.t = t
        self.type_reg = 'ca'

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
    r"""
    Correlation Regularization for the outputs of optical layers.

    This regularizer computes 

    .. math::
        \begin{equation*}
        R(\mathbf{x},\mathbf{y}) = \mu\left\|\mathbf{C_{xx}} - \mathbf{C_{yy}}\right\|_2
        \end{equation*}
    
    where :math:`\mathbf{C_{xx}}` and :math:`\mathbf{C_{yy}}` are the correlation matrices of the input tensors :math:`\mathbf{x}` and :math:`\mathbf{y}`, respectively and `\mu` is a regularization parameter .

    

    """

    def __init__(self, batch_size=128, param=0.001):
        """

        Args:
            batch_size (int): Batch size used for reshaping.
            param (float): Regularization parameter.
        """        
        super(Correlation, self).__init__()
        self.batch_size = batch_size
        self.param = param
        self.type_reg = 'measurements'

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
    r"""
    KL Divergence Regularization for Gaussian Distributions.
    
    Code adapted from [2] Jacome, Roman, Pablo Gomez, and Henry Arguello.
    "Middle output regularized end-to-end optimization for computational imaging." 
    Optica 10.11 (2023): 1421-1431.

    .. math::
        \begin{equation*}
        R(\mathbf{y}) = \text{KL}(p_\mathbf{y},p_\mathbf{z}) = -\frac{1}{2}\sum_{i=1}^{n} \left(1 + \log(\sigma_{\mathbf{y}_i}^2) - \log(\sigma_{\mathbf{z}_i}^2) - \frac{\sigma_{\mathbf{y}_i}^2 + (\mu_{\mathbf{y}_i} - \mu_{\mathbf{z}_i})^2}{\sigma_{\mathbf{z}_i}^2}\right)
        \end{equation*}
    
    where :math:`\mu_{\mathbf{y}_i}` and :math:`\sigma_{\mathbf{y}_i}` are the mean and standard deviation of the input tensor :math:`\mathbf{y}`, respectively, and :math:`\mu_{\mathbf{z}_i}` and :math:`\sigma_{\mathbf{z}_i}` are the target mean and standard deviation, respectively.

    """

    def __init__(self, mean=1e-2, stddev=2.0):
        super(KLGaussian, self).__init__()
        """ 
            Args:
                mean (float): Target mean for the Gaussian distribution.
                stddev (float): Target standard deviation for the Gaussian distribution.
            
            Returns:
                torch.Tensor: KL divergence regularization term.
        """
        self.mean = mean
        self.stddev = stddev
        self.type_reg = 'measurements'

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
            z_log_var - torch.log(self.stddev*torch.ones_like(z_log_var)) - (torch.exp(z_log_var) + torch.pow(z_mean - self.mean, 2)) / (
                    self.stddev ** 2) + 1)
        return kl_loss

class MinVariance(nn.Module):
    r"""
    Minimum Variance Regularization.


    Code adapted from [2] Jacome, Roman, Pablo Gomez, and Henry Arguello.
    "Middle output regularized end-to-end optimization for computational imaging." 
    Optica 10.11 (2023): 1421-1431.

    .. math::
        \begin{equation*}
        R(\mathbf{y}) = \mu\left\|\sigma_{\mathbf{y}}\right\|_2
        \end{equation*}
    
    where :math:`\sigma_{\mathbf{y}}` is the standard deviation of the input tensor :math:`\mathbf{y}`.
    """

    def __init__(self, param=1e-2):
        
        """
        Args:
            param (float): Regularization parameter.
        """   
        super(MinVariance, self).__init__()
        self.param = param
        self.type_reg = 'measurements'
     

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




    
