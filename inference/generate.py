"""Генерация текста с опциональным E8GraphResonator."""

import re
import torch
import torch.nn.functional as F


def generate(model, sp, start_str="who is Lily?", max_tokens=112, temperature=0.5,
             top_k=50, top_p=0.9, repetition_penalty=1.2, repetition_window=50,
             device=None, resonator=None, resonance_strength=0.07, encode_relation_weight=1.0):
    """
    Генерация текста с возможностью использования дискретного резонатора.
    resonator: экземпляр E8GraphResonator (опционально)
    resonance_strength: коэффициент влияния резонанса на логиты
    encode_relation_weight: насколько сильно каждая связь токен→токен записывается в граф
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block_size = model.cfg.block_size
    model.eval()
    if resonator is not None:
        resonator.eval()

    # Кодируем промпт
    prompt_ids = sp.encode(start_str)
    if not prompt_ids:
        prompt_ids = [sp.bos_id() if sp.bos_id() != -1 else sp.unk_id()]

    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # ID специальных токенов
    punct_ids = [pid for p in ':,.!?;' if (pid := sp.piece_to_id(p)) != -1]
    space_id = sp.piece_to_id('▁')
    if space_id == -1:
        space_id = sp.piece_to_id(' ')

    # Подготовка данных для резонатора
    if resonator is not None:
        tok_emb = model.tok_emb.weight
        tok_emb_norm = F.normalize(tok_emb, dim=-1)
        roots_norm = F.normalize(resonator.e8_roots, dim=-1)
        sim = tok_emb_norm @ roots_norm.T
        token_to_root = sim.argmax(dim=-1).to(device)
    else:
        token_to_root = None

    generated_tokens = []
    prev_root_idx = None

    eos_id = sp.eos_id() if sp.eos_id() != -1 else sp.piece_to_id('<|endoftext|>')
    if eos_id == -1:
        eos_id = None

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = context[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[0, -1, :].clone()

            # ДИНАМИЧЕСКАЯ ПАМЯТЬ (резонатор)
            if resonator is not None and token_to_root is not None:
                curr_token = context[0, -1].item()
                curr_root_idx = token_to_root[curr_token].item()

                if prev_root_idx is not None and prev_root_idx != curr_root_idx:
                    resonator.encode_relation(prev_root_idx, curr_root_idx, encode_relation_weight)

                resonance = resonator(curr_root_idx)
                if resonance.dim() > 1:
                    resonance = resonance.squeeze(0)
                resonance_bias = resonance[token_to_root]
                logits += resonance_strength * resonance_bias
                prev_root_idx = curr_root_idx

            # Штрафы за повторения
            past_tokens = context[0, -repetition_window:].tolist()
            for t_idx in set(past_tokens):
                count = past_tokens.count(t_idx)
                logits[t_idx] -= repetition_penalty * count

            if len(context[0]) > 0:
                last_token = context[0, -1].item()
                if last_token in punct_ids:
                    logits[last_token] -= 50.0

            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)

            # Top-K
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(probs, min(top_k, probs.size(-1)))
                probs[probs < v[-1]] = 0.0

            # Top-P
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = 0
                indices_to_remove = sorted_indices[mask]
                probs[indices_to_remove] = 0.0

            probs_sum = probs.sum()
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = torch.zeros_like(probs)
                probs[space_id if space_id != -1 else sp.unk_id()] = 1.0

            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
            token_id = next_token.item()

            # Проверка EOS до печати
            if eos_id is not None and token_id == eos_id:
                break

            generated_tokens.append(token_id)

            # Онлайн-вывод токенов (как в Colab-версии)
            piece = sp.id_to_piece(token_id)
            # Фильтрация служебных токенов
            if piece == '<pad>' or (piece.startswith('<0x') and piece.endswith('>')):
                continue
            if piece == '<0x22>':
                piece = '"'
            elif piece == '<0x0A>':
                piece = '\n'
            elif piece.startswith('▁'):
                piece = ' ' + piece[1:]
            print(piece, end='', flush=True)

    print()  # перевод строки после генерации
    full_text = sp.decode(generated_tokens)
    clean_text = re.split(r'<pad>|<0x[0-9A-Fa-f]+>', full_text)[0]
    return clean_text
