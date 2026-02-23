import torch
import torch.nn as nn
from .attention import E8Attention
from .quantizer import E8Quantizer

class E8TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ... код
        self.quantizer = E8Quantizer(temp=cfg.temp)
        # ...
    
    def forward(self, x):
        # ... код