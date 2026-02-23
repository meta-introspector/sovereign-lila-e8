import torch
from data_utils.streaming_dataset import get_batch_streaming, create_train_val_iterators
from tokenizer.tokenizer_utils import load_tokenizer
from models.model import E8GPT
from config.config import E8Config
from training.checkpoint import save_checkpoint, load_latest_checkpoint
import os

def train(checkpoint_dir="checkpoints", resume=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Загружаем токенайзер
    sp = load_tokenizer()  # или передать путь
    vocab_size = sp.get_piece_size()

    # Конфиг
    cfg = E8Config(vocab_size=vocab_size)
    model = E8GPT(cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    start_step = 0
    if resume:
        start_step = load_latest_checkpoint(model, optimizer, checkpoint_dir, device)

    train_iter, val_iter = create_train_val_iterators()

    # Гиперпараметры
    batch_size = 4
    block_size = cfg.block_size
    total_steps = 150000
    log_every = 200
    save_every = 1000
    gen_every = 1000

    model.train()
    for step in range(start_step + 1, total_steps + 1):
        xb, yb = get_batch_streaming(train_iter, batch_size, block_size, device, sp)
        if xb is None:
            train_iter, _ = create_train_val_iterators()  # пересоздаём
            continue

        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0:
            print(f"Step {step}: loss {loss.item():.4f}")

        if step % save_every == 0:
            save_checkpoint(step, model, optimizer, loss.item(), checkpoint_dir)

    save_checkpoint(total_steps, model, optimizer, loss.item(), checkpoint_dir)