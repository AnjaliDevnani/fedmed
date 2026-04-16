# utils.py
import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Track running average of any scalar (loss, accuracy, etc.)"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Stop training when metric stops improving."""
    def __init__(self, patience=5, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best = None
    
    def __call__(self, metric) -> bool:
        if self.best is None:
            self.best = metric
            return False
        
        improved = (metric > self.best + self.min_delta 
                   if self.mode == 'max'
                   else metric < self.best - self.min_delta)
        
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('val_auc', 0.0)
