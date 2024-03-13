import torch




class L2(torch.nn.Module):
    r"""
        L2 fidelity

        L2 fidelity is defined as the squared L2 norm of the difference between the data and the model prediction.

        .. math::
            \frac{1}{2}||H(x) - y||^2_2

    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H=None):
        """ Computes the L2 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """

        return 1/2*torch.norm( H(x) - y,p=2)**2
    
    def grad(self, x, y, H=None, transform=None):
        x = x.requires_grad_()
        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]



class L1(torch.nn.Module):
    r"""
        L1 fidelity

        L1 fidelity is defined as the L1 norm of the difference between the data and the model prediction.

        .. math::
            ||H(x) - y||_1
    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H):
        """ Computes the L1 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The data to be reconstructed.
            H (function): The forward model.

        Returns:
            torch.Tensor: The L1 fidelity term.
        """
        
        return torch.norm( H(x) - y,p=1)
    
    def grad(self, x, y, H):
        x = x.requires_grad_()

        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]
    


        


