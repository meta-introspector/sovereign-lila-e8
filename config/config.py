from dataclasses import dataclass

@dataclass
class E8Config:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    block_size: int = 512
    dropout: float = 0.05
    e8_scale: float = 0.0266
    bias: bool = False
    e8_strength: float = 0.01
    temp: float = 0.1
    tie_weights: bool = True