import torch
import time
class Training():

    """
    Class for training a neural network model.
    """

    def __init__(self, model, train_loader, optimizer, loss_func, losses_weights, metrics, regularizers, regularization_weights, schedulers = [], callbacks = [], device='cpu'):
        """
        Args:
            model (torch.nn.Module): Neural network model.
            train_loader (torch.utils.data.DataLoader): Training data loader.
            optimizer (torch.optim): Optimizer.
            loss_func (dict): Dictionary of loss functions, format: {"name_loss":function}.
            losses_weights (list): List of weights for each loss function.
            metrics (dict): Dictionary of metrics, format: {"name_metric":function}.
            regularizers (dict): Dictionary of regularizers, format: {"name_regularizer":function}.
            regularization_weights (list): List of weights for each regularizer.
            schedulers (list): List of learning rate schedulers.
            callbacks (list): List of callbacks.
            device (str): Device to use for training.
        """
        self.model = model
        self.train_loader = train_loader
        self.loss_func = loss_func
        self.losses_weights = losses_weights
        self.optimizer = optimizer
        self.metrics = metrics
        self.regularizers = regularizers
        self.regularization_weights = regularization_weights
        self.schedulers = schedulers
        self.callbacks = callbacks
        self.device = device

    def train_one_epoch(self, freq=1, steps_per_epoch = None):
        """
        Train model for one epoch.

        Args:
            freq (int): Frequency for printing training progress.
            steps_per_epoch (int): Number of steps per epoch.
        
        Returns:
            dict: Dictionary of loss values.
        """


        ## Compute time for each batch
        
        start_time = time.time()
        for i, data in enumerate(self.train_loader):
            # Every data instance is an input + outputs pair

            inputs, _ = data
            outputs_gt = inputs.clone()
            inputs  = inputs.to(self.device)
            outputs_gt = outputs_gt.to(self.device)

            # Zero your gradients for every batch!
            self.optimizer.zero_grad(set_to_none=True)

            # Make inference 
            outputs_pred = self.model(inputs)


            final_loss = 0.0
            loss_values = {}#loss_values = { key: 0.0 for key in self.loss_func.keys()}
            for idx, key in enumerate(self.loss_func.keys()):
                res = self.loss_func[key](outputs_pred, outputs_gt) * self.losses_weights[idx]
                loss_values[key] = res
                final_loss += loss_values[key]

            final_loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Keep track of loss



            with torch.no_grad():
                final_metric = 0.0
                metric_values = {}
                for idx, key in enumerate(self.metrics.keys()):
                    res = self.metrics[key](outputs_pred, outputs_gt)
                    metric_values[key] = res
                    final_metric += metric_values[key]
            # Elapsed time
            

            if i % freq == 0:  # print every freq mini-batches
                elapsed_time = time.time() - start_time; start_time = time.time()
                t = elapsed_time / freq; 

                txt_metrics = "".join([f"{key}: {metric_values[key]:.3E}, " for key in metric_values.keys()])
                txt_losses = "".join([f"{key}: {loss_values[key]:.3E}, " for key in loss_values.keys()])
                print(f'  batch {i + 1}/{len(self.train_loader)}, {txt_losses}, {txt_metrics}, time per batch: {t:.1f} [s]')
            if steps_per_epoch != None and i >= steps_per_epoch:
                return loss_values

        
        return loss_values

    def reg_one_epoch(self, verbose = False):
        """
        Weight regularization for one epoch.
        """
        running_reg = 0.
        reg_values = {}
        for idx, key in enumerate(self.regularizers.keys()):
            for p in self.model.parameters():
                if p.requires_grad:
                    reg = self.regularizers[key](p) * self.regularization_weights[idx]
                    reg_values[key] = reg
                    running_reg += reg
        if verbose and len(reg_values) > 0:
            print(f'  regularization loss: {running_reg:.5E}')
        return running_reg


    def fit(self, n_epochs, verbose_reg = True, freq=1, steps_per_epoch = None):
        """
        Train model 
        
        Args:
            n_epochs (int): Number of epochs.
            freq (int): Frequency for printing training progress.
            steps_per_epoch (int): Number of steps per epoch.
        
        Returns:
            dict: Dictionary of loss values.
        """
        start_time = time.time()
        for epoch in range(n_epochs):
            
            print('Epoch {}/{}'.format(epoch + 1, n_epochs))

            self.model.train(True)
            if self.loss_func:
                results_fidelities  = self.train_one_epoch(freq = freq, steps_per_epoch = steps_per_epoch)
            else:
                results_fidelities = {}
            if self.regularizers:
                results_reg = self.reg_one_epoch_with_data(verbose = verbose_reg)
            else:
                results_reg = {}
            results_losses = {**results_fidelities, **results_reg}
            self.model.train(False)

            for s in self.schedulers:
                s.step()

            for c in self.callbacks:
                c.step(self.model, results_losses, epoch)

            elapsed_time = time.time() - start_time; start_time = time.time()
            print('  time per epoch: {:.1f} [s]'.format(elapsed_time))

        return results_losses
    
