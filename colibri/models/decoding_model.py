import torch.nn as nn

class Decoder(nn.Module):

    def __init__(self, model):
        super(Decoder, self).__init__()
        self.model = model
    def forward(self, y, acquisition_model =None):
        if acquisition_model is not None:
            y = acquisition_model(y, type_calculation="backward")
        x = self.model(y)
        return x

    