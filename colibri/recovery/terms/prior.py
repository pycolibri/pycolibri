import torch




class Sparsity(torch.nn.Module):
    '''
        Sparsity prior 
        
        .. math::
        
            g(\mathbf{x}) = \|x\|_1


    '''
    def __init__(self):
        '''
        Args:
            None
        '''
        super(Sparsity, self).__init__()

    def forward(self, x):
        '''
        Compute sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Sparsity term.
        '''
        return torch.norm(x,1)**2
    
    def prox(self, x, _lambda, type="soft"):
        '''
        Compute proximal operator of the sparsity term.

        Args:
            x (torch.Tensor): Input tensor.
            _lambda (float): Regularization parameter.
            type (str): String, it can be "soft" or "hard".
        
        Returns:
            torch.Tensor: Proximal operator of the sparsity term.
        '''
        x = x.requires_grad_()

        if type == 'soft':
            return torch.sign(x)*torch.max(torch.abs(x) - _lambda, torch.zeros_like(x))
        elif type == 'hard':
            return x*(torch.abs(x) > _lambda)
        
        
    

    

