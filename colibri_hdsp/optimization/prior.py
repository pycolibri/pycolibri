import torch




class Sparsity(torch.nn.Module):
    '''
        L2 fidelity 
    
    '''
    def __init__(self):
        super(Sparsity, self).__init__()

    def forward(self, x):
        return torch.norm(x,1)**2
    
    def prox(self, x, type):
        x = x.requires_grad_()

        if type == 'soft':
            return torch.sign(x)*torch.max(torch.abs(x) - self.algo_params['lambda'], torch.zeros_like(x))
        elif type == 'hard':
            return x*(torch.abs(x) > self.algo_params['lambda'])
        
        
    

    

