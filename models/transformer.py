# Класс E8Transformer (без головы). Импортирует E8TransformerLayer

import torch
import torch.nn as nn
from .layer import E8TransformerLayer

class E8Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([E8TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # ... код

    def forward(self, idx, targets=None):
        # ... код