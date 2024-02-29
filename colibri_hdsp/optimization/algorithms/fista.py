import torch 

class Fista(torch.nn.Module):
    '''
    FISTA algorithm for solving the optimization problem
    min_x 1/2||y - Hx||^2 + lambda||x||_1
    '''
    def __init__(self, fidelity, prior, acquistion_model, algo_params):
        super(Fista, self).__init__()
        
        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior
        self.algo_params = algo_params
        
    def forward(self, y, x0=None):
        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        t = 1
        z = x.clone()

        for _ in range(self.algo_params['max_iter']):
            x_old = x.clone()
            z_old = z.clone()
            x = self.prior.prox(z - self.algo_params['alpha']*self.fidelity.grad(x,y,self.acquistion_model), self.algo_params['lambda'])
            t_old = t
            t = (1 + (1 + 4 * t_old ** 2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)

            if torch.norm(x - x_old) < self.tol and torch.norm(z - z_old) < self.tol:
                break

        return x

    

