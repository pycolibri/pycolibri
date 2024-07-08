import torch




class L2(torch.nn.Module):
    r"""
        L2 fidelity

        .. math::
           f(\mathbf{x}) =  \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H=None):
        r""" Computes the L2 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """

        return 1/2*torch.norm( H(x) - y,p=2)**2
    
    def grad(self, x, y, H=None, transform=None):
        r'''
        Compute the gradient of the L2 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Measurements tensor.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()
        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]



class L1(torch.nn.Module):
    r"""
        L1 fidelity

        .. math::
            f(\mathbf{x}) = ||\forwardLinear(\mathbf{x}) - \mathbf{y}||_1
    """
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H):
        r""" Computes the L1 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): Measurements tensor.
            H (function): The forward model.

        Returns:
            torch.Tensor: The L1 fidelity term.
        """
        
        return torch.norm( H(x) - y,p=1)
    
    def grad(self, x, y, H):
        r'''
        Compute the gradient of the L1 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||_1
            

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Measurements tensor.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()

        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]
    


        


