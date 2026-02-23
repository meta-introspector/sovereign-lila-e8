import torch

@torch.jit.script
def jit_e8_distance_matrix(z: Tensor, roots: Tensor) -> Tensor:
    # ... (код функции)

@torch.jit.script
def jit_soft_e8_quantization(distances_sq: Tensor, temp: float, roots: Tensor) -> Tensor:
    # ... (код функции)

@torch.jit.script
def jit_e8_attention_bias(q: Tensor, k: Tensor, head_dirs: Tensor, head_scales: Tensor) -> Tensor:
    # ... (код функции)