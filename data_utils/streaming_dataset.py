# Функция get_batch_streaming и инициализация итераторов.

import torch
import random
from collections import deque
from datasets import load_dataset


def get_batch_streaming(iterator, batch_size, block_size, device, tokenizer, pad_token_id=1, buffer_size=200):
    """Получает батч из стримингового датасета с буфером для перемешивания."""
    x_batch, y_batch = [], []
    buffer = deque()

    # Наполняем буфер
    while len(buffer) < buffer_size:
        try:
            ex = next(iterator)
            buffer.append(ex)
        except StopIteration:
            break

    while len(x_batch) < batch_size:
        if not buffer:
            return None, None
        ex = random.choice(buffer)
        tokens = tokenizer.encode(ex['text'])
        if len(tokens) <= 1:
            continue

        # Если последовательность длиннее block_size+1, берём случайный кусок
        if len(tokens) > block_size + 1:
            start = random.randint(0, len(tokens) - block_size - 1)
            chunk = tokens[start:start + block_size + 1]
        else:
            # Иначе дополняем паддингом до нужной длины
            pad_len = block_size + 1 - len(tokens)
            chunk = tokens + [pad_token_id] * pad_len

        x_batch.append(chunk[:-1])
        y_batch.append(chunk[1:])

    # Обновляем буфер
    try:
        new_ex = next(iterator)
        buffer.append(new_ex)
        buffer.popleft()
    except StopIteration:
        pass

    X = torch.tensor(x_batch, dtype=torch.long, device=device)
    Y = torch.tensor(y_batch, dtype=torch.long, device=device)
    return X, Y


def create_train_val_iterators():
    """Создаёт итераторы для локальных spool данных."""
    from pathlib import Path
    import json
    
    dataset_path = Path("spool_dataset.json")
    
    # Load preprocessed dataset (instant)
    if dataset_path.exists():
        print(f"⚡ Loading preprocessed dataset from {dataset_path}")
        with open(dataset_path) as f:
            texts = json.load(f)
        print(f"✅ Loaded {len(texts)} documents instantly")
    else:
        # Fallback to slow loading
        print("⚠️  No preprocessed dataset. Run: cd preprocessor && cargo run --release")
        print("   Falling back to slow file-by-file loading...")
        filelist_path = Path("spool_filelist.json")
    
    # Load from cached filelist if exists
    if filelist_path.exists():
        print(f"📂 Loading cached filelist from {filelist_path}")
        with open(filelist_path) as f:
            file_paths = json.load(f)
        print(f"✅ Loaded {len(file_paths)} files from cache")
    else:
        print("⚠️  No cached filelist found. Run: python scripts/build_spool_filelist.py")
        print("   Falling back to scanning (this will be slow)...")
        file_paths = []
        extensions = ["*.txt", "*.md", "*.rs", "*.py", "*.lean", "*.nix", "*.toml", "*.el", "*.sh"]
        for path in [Path("/mnt/data1/spool"), Path.home() / "DOCS"]:
            if path.exists():
                for ext in extensions:
                    file_paths.extend([str(f) for f in path.rglob(ext)])
    
    texts = []
    print(f"📖 Reading file contents...")
    for i, fpath in enumerate(file_paths):
        if i % 1000 == 0:
            print(f"   Progress: {i}/{len(file_paths)}")
        try:
            text = Path(fpath).read_text()[:2000]
            if len(text) > 50:
                texts.append({'text': text})
        except:
            pass
    
    # Split 90/10 train/val
    split_idx = int(len(texts) * 0.9)
    train_data = texts[:split_idx]
    val_data = texts[split_idx:]
    
    print(f"✅ Loaded {len(texts)} total documents")
    print(f"   Train: {len(train_data)} | Val: {len(val_data)}")
    print(f"   Split: 90/10")
    print()
    
    def cycle_iterator(data):
        while True:
            random.shuffle(data)
            for item in data:
                yield item
    
    return cycle_iterator(train_data), cycle_iterator(val_data)
