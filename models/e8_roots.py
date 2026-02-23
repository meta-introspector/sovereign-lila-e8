import torch
import math

@torch.jit.script
def _generate_e8_type1_roots() -> torch.Tensor:
    # ... код

@torch.jit.script
def _generate_e8_type2_roots() -> torch.Tensor:
    # ... код

def _verify_e8_mathematical_properties(roots: torch.Tensor) -> None:
    # ... код (можно оставить как есть или убрать проверку)

def _compute_production_e8_roots() -> torch.Tensor:
    # ... код, можно убрать print

# Кэш корней
_E8_ROOTS_CACHE = _compute_production_e8_roots()

def get_e8_roots():
    return _E8_ROOTS_CACHE