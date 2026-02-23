# ==================== КОНФИГУРАЦИЯ ====================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.config import E8Config


# ============================================================
# МАТЕМАТИЧЕСКИ КОРРЕКТНЫЕ КОРНИ СИСТЕМЫ E8 (240 ШТУК)
# ============================================================

@torch.jit.script
def _generate_e8_type1_roots() -> Tensor:
    roots = torch.zeros(112, 8, dtype=torch.float32)
    idx = 0
    for i in range(8):
        for j in range(i + 1, 8):
            for sign_i in [-1.0, 1.0]:
                for sign_j in [-1.0, 1.0]:
                    roots[idx, i] = sign_i
                    roots[idx, j] = sign_j
                    idx += 1
    return roots

@torch.jit.script
def _generate_e8_type2_roots() -> Tensor:
    roots = torch.zeros(128, 8, dtype=torch.float32)
    idx = 0
    for mask in range(256):
        negative_count = 0
        root = torch.zeros(8, dtype=torch.float32)
        for bit in range(8):
            if (mask >> bit) & 1:
                root[bit] = -0.5
                negative_count += 1
            else:
                root[bit] = 0.5
        if negative_count % 2 == 0:
            roots[idx] = root
            idx += 1
    return roots

def _verify_e8_mathematical_properties(roots: Tensor) -> None:
    assert roots.shape == (240, 8), f"Неверная размерность: {roots.shape}"
    norms = torch.norm(roots, dim=1)
    expected_norm = math.sqrt(2.0)
    max_error = torch.abs(norms - expected_norm).max().item()
    assert max_error < 1e-6, f"Нормы неточные! Макс. ошибка: {max_error}"
    unique_roots = torch.unique(roots, dim=0)
    assert unique_roots.shape[0] == 240, f"Дублированные корни! Уникальных: {unique_roots.shape[0]}"
    center_deviation = torch.norm(roots.sum(dim=0)).item()
    assert center_deviation < 1e-4, f"Не центрировано: {center_deviation}"

def _compute_production_e8_roots() -> Tensor:
    """
    Генерация математически корректных корней E8 один раз за процесс.
    Без лишних print, проверка свойств остаётся.
    """
    type1_roots = _generate_e8_type1_roots()
    type2_roots = _generate_e8_type2_roots()
    all_roots = torch.cat([type1_roots, type2_roots], dim=0)
    _verify_e8_mathematical_properties(all_roots)
    return all_roots

# ============================================================
# JIT-КОМПИЛИРОВАННЫЕ CORE ОПЕРАЦИИ
# ============================================================

@torch.jit.script
def jit_e8_distance_matrix(z: Tensor, roots: Tensor) -> Tensor:
    z_norms_sq = (z * z).sum(dim=-1, keepdim=True)
    cross_terms = torch.matmul(z, roots.T)
    root_norms_sq = (roots * roots).sum(dim=-1)
    distances_sq = z_norms_sq - 2.0 * cross_terms + root_norms_sq
    return distances_sq

@torch.jit.script
def jit_soft_e8_quantization(distances_sq: Tensor, temp: float, roots: Tensor) -> Tensor:
    weights = torch.softmax(-distances_sq / temp, dim=-1)
    return torch.matmul(weights, roots)

@torch.jit.script
def jit_e8_attention_bias(q: Tensor, k: Tensor, head_dirs: Tensor, head_scales: Tensor) -> Tensor:
    B, H, T, D = q.shape
    if D >= 8:
        q_e8 = q[:, :, :, :8]
        k_e8 = k[:, :, :, :8]
        q_proj = torch.einsum('bhtd,hd->bht', q_e8, head_dirs)
        k_proj = torch.einsum('bhtd,hd->bht', k_e8, head_dirs)
        geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
        return head_scales.view(1, -1, 1, 1) * geo_bias
    else:
        return torch.zeros(B, H, T, T, device=q.device, dtype=q.dtype)

# ==================== КОРНИ E8 (ОДИН РАЗ НА ПРОЦЕСС) ====================
_e8_roots_raw = _compute_production_e8_roots()
e8_roots = torch.nn.functional.normalize(_e8_roots_raw.to(torch.float32), p=2, dim=1)

# ==================== E8 КВАНТИЗАТОР ====================
class E8Quantizer(nn.Module):
    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp
        self.register_buffer('roots', e8_roots)
        self.register_buffer('roots_norm_sq', (e8_roots ** 2).sum(dim=1))

    def forward(self, x):
        roots = self.roots
        roots_norm_sq = self.roots_norm_sq
        x_norm_sq = (x ** 2).sum(dim=-1, keepdim=True)
        cross = x @ roots.T
        distances = x_norm_sq - 2 * cross + roots_norm_sq
        weights = F.softmax(-distances / self.temp, dim=-1)
        quantized = weights @ roots
        return x + (quantized - x).detach()

    def hard_quantize(self, x):
        distances = torch.cdist(x, self.roots)
        indices = torch.argmin(distances, dim=-1)
        return self.roots[indices]

# ==================== E8 LINEAR ====================
class E8Linear(nn.Module):
    def __init__(self, in_features: int, cfg: E8Config):
        super().__init__()
        self.cfg = cfg
        self.in_features = in_features
        if cfg.tie_weights:
            self.weight = nn.Parameter(torch.randn(in_features, 8) * 0.02)
            self.proj = lambda x: F.linear(x, self.weight.t(), None)
            self.unproj = lambda x: F.linear(x, self.weight, None)
        else:
            self.proj = nn.Linear(in_features, 8, bias=cfg.bias)
            self.unproj = nn.Linear(8, in_features, bias=cfg.bias)
        self.quantizer = E8Quantizer(cfg.temp)
        self.alpha = nn.Parameter(torch.tensor(cfg.e8_strength))

    def forward(self, x):
        z = self.proj(x)
        z_quantized = self.quantizer(z)
        z_mixed = z + self.alpha * (z_quantized - z)
        return self.unproj(z_mixed)


class E8GraphResonator(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.d_model = d_model
        # Инициализируем фиксированные ориентиры - корни E8
        # (используем твой генератор корней из предыдущих шагов)
        self.register_buffer("e8_roots", self._get_e8_basis(d_model))
        
        # Матрица связей (Ассоциативная память)
        # В идеале это тензор 4-го ранга, но для PoC возьмем обучаемую проекцию
        self.reasoning_gate = nn.Parameter(torch.randn(d_model, d_model) * 0.02)

    def _get_e8_basis(self, d_model):
        # 1. Генерируем "сырые" корни
        raw_roots = self._generate_240_roots() # (240, 8)
        
        # 2. НОРМАЛИЗАЦИЯ (Критически важно!)
        # Приводим длину каждого корня к 1.0. 
        # Это убирает "черные дыры" в пространстве смыслов.
        raw_roots = torch.nn.functional.normalize(raw_roots, p=2, dim=1)
        
        # 3. ПРАВИЛЬНАЯ ПРОЕКЦИЯ (Масштабирование по входу, а не по выходу)
        # Мы делим на sqrt(8), потому что проецируем ИЗ 8 измерений.
        # Это сохраняет дисперсию сигнала.
        proj = torch.randn(8, d_model) / (8**0.5) 
        
        return torch.matmul(raw_roots, proj) # (240, d_model)

    def _generate_240_roots(self):
        roots = []
        # Тип 1: (±1, ±1, 0^6)
        for i in range(8):
            for j in range(i + 1, 8):
                for s1 in [-1.0, 1.0]:
                    for s2 in [-1.0, 1.0]:
                        r = [0.0]*8
                        r[i], r[j] = s1, s2
                        roots.append(r)
        # Тип 2: (±0.5)^8 (четное кол-во минусов)
        for i in range(256):
            r = [0.5 if not ((i >> b) & 1) else -0.5 for b in range(8)]
            if sum(1 for x in r if x < 0) % 2 == 0:
                roots.append(r)
        return torch.tensor(roots, dtype=torch.float32)


    def encode_relation(self, node_a_idx, node_b_idx, relation_weight=1.0):
        """
        Записывает связь между двумя узлами графа как геометрический сдвиг в пространстве E8.
        relation_weight: насколько сильно каждая новая связь влияет на reasoning_gate.
        """
        root_a = self.e8_roots[node_a_idx % 240]
        root_b = self.e8_roots[node_b_idx % 240]
        relation = torch.outer(root_a, root_b)
        with torch.no_grad():
            self.reasoning_gate.add_(relation * relation_weight)

    def forward(self, query_node_idx):
        """
        Поиск по графу: Подаем узел, получаем резонанс со связанными узлами
        """
        # 1. Берем вектор запроса (корень E8)
        query_vec = self.e8_roots[query_node_idx % 240].unsqueeze(0)
        
        # 2. Пропускаем через "Геометрические веса" (наш граф)
        # Это эквивалентно поиску пути в группе Ли
        latent_response = torch.matmul(query_vec, self.reasoning_gate)
        
        # 3. Резонанс: На какие корни E8 похож результат?
        # Это и есть "выдача" из графа знаний
        scores = torch.matmul(F.normalize(latent_response, dim=-1), 
                              F.normalize(self.e8_roots, dim=-1).t())
        
        return scores # Возвращает вероятности связей со всеми 240 корнями

# ==================== E8 ATTENTION ====================
class E8Attention(nn.Module):
    def __init__(self, cfg: E8Config):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer('head_dirs', e8_roots[:cfg.n_heads])
        self.head_scales = nn.Parameter(torch.zeros(cfg.n_heads))
        self.register_buffer("causal_mask", torch.tril(torch.ones(1, 1, cfg.block_size, cfg.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, float('-inf'))
        if self.head_dim >= 8:
            q_e8 = q[..., :8]
            k_e8 = k[..., :8]
            q_proj = torch.einsum('bhtd,hd->bht', q_e8, self.head_dirs)
            k_proj = torch.einsum('bhtd,hd->bht', k_e8, self.head_dirs)
            geo_bias = q_proj.unsqueeze(-1) * k_proj.unsqueeze(-2)
            scores = scores + self.head_scales.view(1, -1, 1, 1) * geo_bias
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
       
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        self.attn_weights = attn  #self.attn_weights = attn #добавил для теста
        return self.out(out)

# ==================== E8 TRANSFORMER LAYER ====================
class E8TransformerLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_attn = nn.LayerNorm(cfg.d_model)
        self.attn = E8Attention(cfg)
        self.ln_res = nn.LayerNorm(cfg.d_model)
        self.to_e8 = nn.Linear(cfg.d_model, 8, bias=False)
        self.from_e8 = nn.Linear(8, cfg.d_model, bias=False)
        self.quantizer = E8Quantizer(temp=cfg.temp)
        self.e8_scale = nn.Parameter(torch.tensor([cfg.e8_strength]))
        # Добавляем резонатор после attention (можно и после FFN)
        #self.resonator = E8ResonatorLayer(cfg.d_model, temperature=10.0, alpha_init=0.1)
        self.ln_ffn = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln_attn(x))
        # Применяем резонатор (можно поставить до или после FFN)
        #x = self.resonator(x)
        res_in = self.ln_res(x)
        e8_space = self.to_e8(res_in)
        res_feat = self.quantizer(e8_space)
        if self.training:
            res_feat = res_feat * (1.0 + 0.001 * torch.randn_like(res_feat))
        res_out = self.from_e8(res_feat)
        if self.training:
            res_out = res_out + torch.randn_like(res_out) * 0.005
        x = x + self.e8_scale * res_out
        x = x + self.ffn(self.ln_ffn(x))
        return x

# ==================== E8 TRANSFORMER (CORE, БЕЗ HEAD) ====================
class E8Transformer(nn.Module):
    def __init__(self, cfg: E8Config):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList([E8TransformerLayer(cfg) for _ in range(cfg.n_layers)])
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.04)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        if idx.dim() == 3:
            b, t, _ = idx.size()
            x = idx
        else:
            b, t = idx.size()
            x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x

# ==================== E8 GPT (ПОЛНАЯ МОДЕЛЬ С HEAD) ====================
class E8GPT(nn.Module):
    def __init__(self, cfg: E8Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.d_model))
        self.core = E8Transformer(cfg)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, idx, targets=None):
        if idx.dim() > 2:
            idx = idx.squeeze()
        b, t = idx.size()
        x = self.tok_emb(idx) + self.pos_emb[:, :t, :]
        hidden_states = self.core(x)
        logits = self.head(hidden_states)
        loss = None
        if targets is not None:
            if targets.dim() > 2:
                targets = targets.squeeze()
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx