import torch
import random
import numpy as np

__all__ = ['str2bool', 'str2list', 'InfIterator', 'set_seed']

def str2list(v):
    return v.split(',')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

class InfIterator(object):
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

    def __len__(self):
        return len(self.iterable)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Make operations deterministic (potentially at the cost of performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

