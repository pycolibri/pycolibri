import os
import random
import numpy as np
import torch

from colibri import models

__all__ = ['models']

from colibri import data

__all__ += ['data']

from colibri import recovery

__all__ += ['recovery']

from colibri import optics

__all__ += ['optics']

from colibri import misc

__all__ += ['misc']

from colibri import metrics

__all__ += ['metrics']



# def seed_everything(seed=42):

#     if seed<=0:
#         raise ValueError("Seed must be a positive integer")
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

