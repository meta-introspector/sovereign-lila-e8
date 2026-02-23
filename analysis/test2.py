
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import os

# Предполагаем, что у вас уже есть:
# model - ваша обученная модель E8GPT
# val_iter - итератор по validation датасету TinyStories
# get_batch_streaming - функция для получения батча
# device - устройство (cuda/cpu)

# Если модель нужно загрузить из чекпоинта, раскомментируйте:
# checkpoint = torch.load('path_to_best_checkpoint.pt')
# model.load_state_dict(checkpoint['model_state_dict'])
# model.to(device)
# model.eval()

# Validation stream (отдельный сплит)
val_dataset = load_dataset("roneneldan/TinyStories", streaming=True, split="validation")
val_iter = iter(val_dataset)


# ============================================================
# 1. СБОР ВСЕХ ЗНАЧЕНИЙ head_scales ПО СЛОЯМ
# ============================================================

def collect_head_scales(model):
    """Собирает параметры head_scales из всех слоёв модели."""
    data = []
    for name, param in model.named_parameters():
        if 'head_scales' in name:
            # Извлекаем номер слоя из имени (предполагается формат core.layers.X.attn.head_scales)
            parts = name.split('.')
            layer_idx = int(parts[2])  # core.layers.2.attn.head_scales -> 2
            values = param.detach().cpu().numpy().flatten()
            for head_idx, val in enumerate(values):
                data.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'beta': val
                })
    return pd.DataFrame(data)

print("Собираем head_scales...")
df_beta = collect_head_scales(model)
print(f"Собрано {len(df_beta)} записей")
print(df_beta.head())

# Сохраняем в CSV
df_beta.to_csv('head_scales_all_layers.csv', index=False)
print("Данные сохранены в head_scales_all_layers.csv")

# ============================================================
# 2. СВОДНАЯ ГИСТОГРАММА
# ============================================================

plt.figure(figsize=(10, 6))
plt.hist(df_beta['beta'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('β (head_scales)')
plt.ylabel('Frequency')
plt.title('Distribution of β across all layers and heads')
plt.grid(True, alpha=0.3)
plt.savefig('beta_histogram_all.png', dpi=150)
plt.show()

# Также можно построить по слоям отдельно
layers = sorted(df_beta['layer'].unique())
fig, axes = plt.subplots(len(layers), 1, figsize=(8, 3*len(layers)), sharex=True)
if len(layers) == 1:
    axes = [axes]
for i, layer in enumerate(layers):
    data_layer = df_beta[df_beta['layer'] == layer]['beta']
    axes[i].hist(data_layer, bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_ylabel(f'Layer {layer}')
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('β')
plt.suptitle('β distribution per layer')
plt.tight_layout()
plt.savefig('beta_histogram_per_layer.png', dpi=150)
plt.show()

# ============================================================
# 3. ФУНКЦИЯ ДЛЯ ВЫЧИСЛЕНИЯ LOSS НА ВАЛИДАЦИИ
# ============================================================

def evaluate_loss(model, val_iter, num_batches=50, device='cuda', disable_geometry=False):
    """
    Оценивает средний loss на заданном количестве батчей из validation.
    Если disable_geometry=True, временно зануляет все head_scales.
    """
    model.eval()
    original_betas = {}  # сохраним оригиналы для восстановления
    
    if disable_geometry:
        # Сохраняем и зануляем head_scales
        for name, param in model.named_parameters():
            if 'head_scales' in name:
                original_betas[name] = param.data.clone()
                param.data.zero_()
    
    total_loss = 0.0
    count = 0
    
    # Создаём копию итератора, чтобы не исчерпать оригинальный
    # Если val_iter уже использовался, лучше пересоздать его из датасета
    # Предположим, у вас есть функция get_val_batch, которая даёт следующий батч
    
    # Простейший способ: использовать тот же итератор, но если он закончится, пересоздать
    local_iter = val_iter  # будем использовать напрямую, но осторожно
    
    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc=f"Evaluating {'without' if disable_geometry else 'with'} geometry"):
            try:
                xb, yb = get_batch_streaming(local_iter, batch_size=4, block_size=512, device=device)
                if xb is None:
                    # Если итератор исчерпан, пересоздаём
                    local_iter = iter(val_dataset)  # нужно, чтобы val_dataset был определён
                    xb, yb = get_batch_streaming(local_iter, batch_size=4, block_size=512, device=device)
            except StopIteration:
                local_iter = iter(val_dataset)
                xb, yb = get_batch_streaming(local_iter, batch_size=4, block_size=512, device=device)
            
            logits, loss = model(xb, yb)
            total_loss += loss.item()
            count += 1
    
    avg_loss = total_loss / count
    
    # Восстанавливаем исходные значения, если зануляли
    if disable_geometry:
        for name, param in model.named_parameters():
            if name in original_betas:
                param.data.copy_(original_betas[name])
    
    return avg_loss

# ============================================================
# 4. СРАВНЕНИЕ LOSS: С ГЕОМЕТРИЕЙ И БЕЗ
# ============================================================

print("Оценка loss с геометрией...")
loss_with_geo = evaluate_loss(model, val_iter, num_batches=100, device=device, disable_geometry=False)
print(f"Loss with geometry: {loss_with_geo:.4f}")

print("Оценка loss без геометрии (β=0)...")
loss_without_geo = evaluate_loss(model, val_iter, num_batches=100, device=device, disable_geometry=True)
print(f"Loss without geometry: {loss_without_geo:.4f}")

print(f"Разница: {loss_without_geo - loss_with_geo:.4f} (без геометрии хуже на эту величину)")

# Сохраняем результаты в текстовый файл
with open('comparison_results.txt', 'w') as f:
    f.write(f"Loss with geometry: {loss_with_geo:.6f}\n")
    f.write(f"Loss without geometry: {loss_without_geo:.6f}\n")
    f.write(f"Difference (without - with): {loss_without_geo - loss_with_geo:.6f}\n")

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ ATTENTION MAPS (ДЛЯ ПРИМЕРА)
# ============================================================

def plot_attention_map(model, layer_idx=0, head_idx=0, device='cuda'):
    """
    Берёт один батч из validation, прогоняет через модель и показывает attention scores
    для указанного слоя и головы (среднее по батчу или первый элемент).
    """
    model.eval()
    
    # Получаем один батч
    xb, yb = get_batch_streaming(val_iter, batch_size=1, block_size=128, device=device)
    if xb is None:
        # пересоздаём итератор и пробуем снова
        #global val_iter
        #val_iter = iter(val_dataset)
        xb, yb = get_batch_streaming(val_iter, batch_size=1, block_size=128, device=device)
    
    # Нам нужно получить attention scores до softmax (или после)
    # Для этого мы должны сделать forward с возвратом attention weights.
    # Предположим, что у вас в модели есть возможность вернуть attention.
    # Если нет, можно временно модифицировать код forward, чтобы он возвращал attn weights.
    # В данном примере я покажу, как получить их, если у вас есть доступ к внутренностям.
    
    # Простой способ: если ваш слой E8Attention хранит attn weights после forward,
    # можно добавить в модель hook или просто модифицировать forward, чтобы он возвращал их.
    # Для демонстрации я создам функцию, которая проходит по слоям и извлекает attn.
    
    def get_attention_scores(model, xb):
        # Это упрощённая версия, предполагающая, что в слоях есть поле attn_weights,
        # которое заполняется в forward.
        # В вашем коде E8Attention сейчас не сохраняет attn, нужно добавить:
        # self.attn_weights = attn  # в конце forward
        # Тогда можно будет их извлечь.
        
        _ = model(xb)
        attn_maps = []
        for layer in model.core.layers:  # предполагаем, что model.core - это E8Transformer
            if hasattr(layer.attn, 'attn_weights'):
                attn_maps.append(layer.attn.attn_weights.detach().cpu())
        return attn_maps
    
    # ВНИМАНИЕ: если у вас нет сохранения attn_weights, нужно добавить в класс E8Attention:
    # в конце forward: self.attn_weights = attn (после softmax)
    # Тогда следующий код сработает.
    
    try:
        attn_maps = get_attention_scores(model, xb)
    except AttributeError:
        print("Модель не сохраняет attention weights. Добавьте self.attn_weights = attn в forward.")
        return
    
    # Выбираем нужный слой и голову
    if layer_idx >= len(attn_maps):
        print(f"Слой {layer_idx} вне диапазона (всего {len(attn_maps)} слоёв)")
        return
    
    attn = attn_maps[layer_idx]  # shape [batch, heads, seq_len, seq_len]
    if head_idx >= attn.size(1):
        print(f"Голова {head_idx} вне диапазона (всего {attn.size(1)} голов)")
        return
    
    # Берём первую картинку из батча
    attn_map = attn[0, head_idx].numpy()  # [seq_len, seq_len]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(attn_map, cmap='viridis')
    plt.title(f'Attention Map - Layer {layer_idx}, Head {head_idx}')
    plt.xlabel('Key position')
    plt.ylabel('Query position')
    plt.savefig(f'attention_layer{layer_idx}_head{head_idx}.png', dpi=150)
    plt.show()

# Пример использования: визуализируем attention для первого слоя и первой головы
print("Генерируем attention map для примера...")
plot_attention_map(model, layer_idx=0, head_idx=0, device=device)

# Если хотите, можно сделать цикл по нескольким головам или слоям.

print("Анализ завершён. Все файлы сохранены.")