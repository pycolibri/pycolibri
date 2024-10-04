import torch

from .transforms import DCT2D


class Prior(torch.nn.Module):
    r"""
        Base class for prior terms.
    """
    def __init__(self):
        super(Prior, self).__init__()

    def forward(self, x):
        r"""
        Compute prior term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Prior term.
        """
        raise NotImplementedError

    def prox(self, x, _lambda):
        r"""
        Compute proximal operator of the prior term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
        
        Returns:
            torch.Tensor: Proximal operator of the prior term.
        """
        raise NotImplementedError

    def reset(self):
        r"""
        Reset prior term.
        """
        pass

class Sparsity(Prior):
    r"""
        Sparsity prior 
        
        .. math::
        
            g(\mathbf{x}) = \| \transform \textbf{x}\|_1
        
        where :math:`\transform` is the sparsity basis and :math:`\textbf{x}` is the input tensor.

    """
    def __init__(self, basis=None,type="soft"):
        r"""
        Args:
            basis (str): Basis function. 'dct', 'None'. Default is None.
        """

        if basis == 'dct':
            self.transform = DCT2D()
        else:
            self.transform = None

        self.type = type
        
        super(Sparsity, self).__init__()


    def forward(self, x):
        r"""
        Compute sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Sparsity term.
        """
        
        x = self.transform.forward(x)
        return torch.norm(x, 1)**2
    
    
    def prox(self, x, _lambda, type="soft"):
        r"""
        Compute proximal operator of the sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
            type (str): String, it can be "soft" or "hard".
        
        Returns:
            torch.Tensor: Proximal operator of the sparsity term.
        """
        
        x = x.requires_grad_()
        x = self.transform.forward(x)

        if type == 'soft':
            x = torch.sign(x)*torch.max(torch.abs(x) - _lambda, torch.zeros_like(x))
        elif type == 'hard':
            x = x*(torch.abs(x) > _lambda)
        
        x = self.transform.inverse(x)
        return x
        
    def transform(self, x):
        
        if self.transform is not None:
            return self.transform.forward(x)
        else:
            return x
    
    def inverse(self, x):
        
        if self.transform is not None:
            return self.transform.inverse(x)
        else:
            return x
        
    
        
        
    

    

