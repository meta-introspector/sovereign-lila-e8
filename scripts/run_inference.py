import argparse
import os
import sys
import torch

# Добавляем корень проекта в path для импортов
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import E8GPT, E8GraphResonator
from tokenizer.tokenizer_utils import load_tokenizer
from inference.generate import generate
from config.config import E8Config

def main():
    parser = argparse.ArgumentParser(description="LILA-E8 inference: generate text from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--prompt", type=str, default="who is Lily?", help="Start prompt")
    parser.add_argument("--max_tokens", type=int, default=112, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature (0.1-2.0)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling (0=disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P (nucleus) sampling (0=disabled)")
    parser.add_argument("--repetition_penalty", type=float, default=1.4, help="Penalty for repeated tokens")
    parser.add_argument("--repetition_window", type=int, default=50, help="Window for repetition penalty")
    parser.add_argument("--no_resonator", action="store_true", help="Disable E8GraphResonator")
    parser.add_argument("--resonance_strength", type=float, default=0.07, help="Resonator: how strongly bias affects logits")
    parser.add_argument("--encode_relation_weight", type=float, default=1.0, help="Resonator: how strongly each token→token relation is written to graph")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sp = load_tokenizer()
    cfg = E8Config(vocab_size=sp.get_piece_size())
    model = E8GPT(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    resonator = None if args.no_resonator else E8GraphResonator(d_model=cfg.d_model).to(device)

    generate(
        model, sp,
        start_str=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p if 0 < args.top_p < 1 else None,
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=device,
        resonator=resonator,
        resonance_strength=args.resonance_strength,
        encode_relation_weight=args.encode_relation_weight,
    )
    # Текст уже выводится потоково внутри generate()

if __name__ == "__main__":
    main()