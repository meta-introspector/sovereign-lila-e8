import os
import torch


def save_checkpoint(step, model, optimizer, loss, checkpoint_dir, is_best=False):
    """Сохраняет чекпоинт в указанную директорию."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = 'best_model.pt' if is_best else f'checkpoint_step_{step}.pt'
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': model.cfg,
    }, path)
    print(f'💾 Чекпоинт сохранён: {path} (loss={loss:.4f})')


def load_latest_checkpoint(model, optimizer, checkpoint_dir, device):
    """Загружает последний чекпоинт или best_model.pt. Возвращает шаг с которого продолжить."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if not files:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'🔄 Загружен best_model.pt, шаг {checkpoint["step"]}, loss {checkpoint["loss"]:.4f}')
            return checkpoint['step']
        print('🆕 Чекпоинтов нет, начинаем с нуля.')
        return 0
    steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
    latest_step = max(steps)
    latest_file = f'checkpoint_step_{latest_step}.pt'
    path = os.path.join(checkpoint_dir, latest_file)
    print(f'📂 Loading checkpoint: {path}')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'🔄 Загружен чекпоинт: шаг {latest_step}, loss {checkpoint["loss"]:.4f}')
    return latest_step
