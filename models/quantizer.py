import torch
import torch.nn as nn
import torch.nn.functional as F
from .e8_roots import get_e8_roots

class E8Quantizer(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        roots = get_e8_roots()
        self.register_buffer('roots', roots)
        self.register_buffer('roots_norm_sq', (roots ** 2).sum(dim=1))

    def forward(self, x):
        # ... код forward (без print)
    
    def hard_quantize(self, x):
        # ... код