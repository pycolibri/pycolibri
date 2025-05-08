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
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """
        r = H(x) - y
        r = r.reshape(r.shape[0],-1)
        return 1/2*torch.norm(r,p=2,dim=1)**2
        # return 1/2*torch.norm( H(x) - y,p=2,)**2
    
    def grad(self, x, y, H=None, transform=None):
        r'''
        Compute the gradient of the L2 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()
        norm = self.forward(x,y,H)
 
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]


class L1(torch.nn.Module):
    r"""
        L1 fidelity

        .. math::
            f(\mathbf{x}) = ||\forwardLinear(\mathbf{x}) - \mathbf{y}||_1
    """
    def __init__(self):
        super(L1, self).__init__()

    def forward(self, x, y, H=None):
        r""" Computes the L2 fidelity term.

        Args:
            x (torch.Tensor): The image to be reconstructed.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function, optional): The forward model. Defaults to None.

        Returns:
            torch.Tensor: The L2 fidelity term.
        """
        r = H(x) - y
        r = r.reshape(r.shape[0],-1)
        return 1/2*torch.norm(r,p=1,dim=1)**2
        # return 1/2*torch.norm( H(x) - y,p=2,)**2
    
    def grad(self, x, y, H=None, transform=None):
        r'''
        Compute the gradient of the L2 fidelity term.

        .. math::
            \nabla f(\mathbf{x}) = \nabla \frac{1}{2}||\forwardLinear(\mathbf{x}) - \mathbf{y}||^2_2

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): The measurement data to be reconstructed.
            H (function): Forward model.   

        Returns:
            torch.Tensor: Gradient of the L1 fidelity term. 
        '''
        x = x.requires_grad_()
        norm = self.forward(x,y,H)
 
        return torch.autograd.grad(norm, x, create_graph=True, grad_outputs=torch.ones_like(norm))[0]
