# ========================
# 1. МОНТИРУЕМ GOOGLE DRIVE (для чекпоинтов)
# ========================
from google.colab import drive
drive.mount('/content/drive')

checkpoint_dir = '/content/drive/MyDrive/e8_tinystories_checkpoints3'
os.makedirs(checkpoint_dir, exist_ok=True)

def save_checkpoint(step, model, optimizer, loss, is_best=False):
    """Сохраняет чекпоинт на Google Drive."""
    filename = 'best_model.pt' if is_best else f'checkpoint_step_{step}.pt'
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': scheduler.state_dict(),  # добавить
        'loss': loss,
        'config': cfg,  # сохраняем гиперпараметры
    }, path)
    print(f'💾 Чекпоинт сохранён: {path} (loss={loss:.4f})')

def load_latest_checkpoint(model, optimizer):
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_step_')]
    if not files:
        # попробуем загрузить best_model.pt
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)  # <--- ДОБАВЬ weights_only=False
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #if 'scheduler_state_dict' in checkpoint:
              #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f'🔄 Загружен best_model.pt, шаг {checkpoint["step"]}, loss {checkpoint["loss"]:.4f}')
            return checkpoint['step']
        print('🆕 Чекпоинтов нет, начинаем с нуля.')
        return 0
    steps = [int(f.split('_')[-1].split('.')[0]) for f in files]
    latest_step = max(steps)

    latest_file = f'checkpoint_step_{latest_step}.pt'

    path = os.path.join(checkpoint_dir, latest_file)
    checkpoint = torch.load(path, map_location=device, weights_only=False)  # <--- ЗДЕСЬ ТОЖЕ
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'🔄 Загружен чекпоинт: шаг {latest_step}, loss {checkpoint["loss"]:.4f}')
    return latest_step