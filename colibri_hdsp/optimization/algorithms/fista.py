import torch 

class Fista(torch.nn.Module):
    '''
    FISTA algorithm for solving the optimization problem
    min_x 1/2||y - Hx||^2 + lambda||x||_1
    '''
    def __init__(self, fidelity, prior, acquistion_model, algo_params, transform):
        super(Fista, self).__init__()
        
        self.fidelity = fidelity
        self.acquistion_model = acquistion_model
        self.prior = prior
        self.algo_params = algo_params
        self.transform = transform

        self.H = lambda x: self.acquistion_model.forward(self.transform.inverse(x))
        self.tol = 1e-4
        
    def forward(self, y, x0=None):
        
        if x0 is None:
            x0 = torch.zeros_like(y)

        x = x0
        t = 1
        z = x.clone()

        for i in range(self.algo_params['max_iter']):
            x_old = x.clone()
            z_old = z.clone()

            # gradient step
            x = z - self.algo_params['alpha']*self.fidelity.grad(z, y, self.H)

            # proximal step
            x = self.prior.prox(x, self.algo_params['lambda'])
            
            # FISTA step
            t_old = t
            t = (1 + (1 + 4 * t_old ** 2) ** 0.5) / 2
            z = x + ((t_old - 1) / t) * (x - x_old)


            # print("Iter: ", i, "Norm x: ", torch.norm(x - x_old), "Norm z: ", torch.norm(z - z_old))

            # if torch.norm(x - x_old) < self.tol and torch.norm(z - z_old) < self.tol:
            #     break
        
        x_hat = self.transform.inverse(x)
        return x_hat

    

