import torch




class L2(torch.nn.Module):
    '''
        L2 fidelity 
    
    '''
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H=None):
        return 1/2*torch.norm( H(x) - y,p=2)**2
    
    def grad(self, x, y, H=None, transform=None):
        x = x.requires_grad_()
        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]



class L1(torch.nn.Module):
    
    '''
        L1 fidelity 
    
    '''
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, H):
        return torch.norm( H(x) - y,p=1)
    
    def grad(self, x, y, H):
        x = x.requires_grad_()

        return torch.autograd.grad(self.forward(x,y, H), x, create_graph=True)[0]
    


        


