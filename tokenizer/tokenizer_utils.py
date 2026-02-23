import os
import sentencepiece as spm
import requests

def load_tokenizer(model_path=None):
    """Загружает токенайзер. Если model_path не указан, ищет в vocab/ и корне проекта."""
    if model_path is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Порядок поиска: vocab/e8_morpheme_.model, vocab/e8_morpheme.model, корень/e8_morpheme.model
        candidates = [
            os.path.join(base, "vocab", "e8_morpheme_.model"),
            os.path.join(base, "vocab", "e8_morpheme.model"),
            os.path.join(base, "e8_morpheme.model"),
        ]
        for p in candidates:
            if os.path.exists(p):
                model_path = p
                break
        else:
            model_path = os.path.join(base, "e8_morpheme.model")  # для обучения
    if not os.path.exists(model_path):
        # Обучаем на Shakespeare (как в Colab)
        base_dir = os.path.dirname(model_path)
        os.makedirs(base_dir, exist_ok=True)
        print("Токенайзер не найден. Обучаем на Shakespeare...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        input_file = os.path.join(base_dir, "input_text.txt")
        with open(input_file, "w", encoding="utf-8") as f:
            f.write(text)
        model_prefix = os.path.join(base_dir, "e8_morpheme")
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=2048,
            model_type='bpe',
            character_coverage=1.0,
            byte_fallback=True,
            unk_id=0, pad_id=1, bos_id=2, eos_id=3
        )
        os.remove(input_file)
    sp = spm.SentencePieceProcessor(model_file=model_path)
    return sp

def encode(text, sp):
    return sp.encode(text)

def decode(tokens, sp):
    return sp.decode(tokens)