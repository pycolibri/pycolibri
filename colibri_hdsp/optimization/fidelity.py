import torch




class L2(torch.nn.Module):
    '''
        L2 fidelity 
    
    '''
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, acquistion_model=None):
        return 1/2*torch.norm(acquistion_model.forward(x) - y,p=2)**2
    
    def grad(self, x, y, acquistion_model=None, transform=None):
        x = x.requires_grad_()
        return torch.autograd.grad(self.forward(x,y,acquistion_model), x, create_graph=True)[0]



class L1(torch.nn.Module):
    
    '''
        L1 fidelity 
    
    '''
    def __init__(self):
        super(L2, self).__init__()

    def forward(self, x, y, acquistion_model):
        return torch.norm(acquistion_model.forward(x) - y,p=1)
    
    def grad(self, x, y, acquistion_model):
        x = x.requires_grad_()

        return torch.autograd.grad(self.forward(x,y,acquistion_model), x, create_graph=True)[0]
    


        


