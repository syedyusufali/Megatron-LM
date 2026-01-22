#!/usr/bin/env python3
"""
Text Generation with Trained GPT Model
=======================================
Load a checkpoint and generate text.

Usage:
    python local_training/generate.py --prompt "The meaning of life is"
    python local_training/generate.py --prompt "Once upon a time" --max_tokens 200
"""

import os
import sys
import torch
import torch.nn as nn
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the model architecture from train_simple
from local_training.train_simple import SimpleGPT, SimpleGPTConfig


def load_tokenizer():
    """Load GPT-2 tokenizer."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        pass

    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        # Make it compatible with tiktoken interface
        class TokenizerWrapper:
            def __init__(self, tok):
                self.tok = tok
            def encode(self, text):
                return self.tok.encode(text)
            def decode(self, tokens):
                return self.tok.decode(tokens)
        return TokenizerWrapper(tokenizer)
    except ImportError:
        print("Please install tiktoken or transformers:")
        print("  pip install tiktoken")
        sys.exit(1)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use defaults
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        config = SimpleGPTConfig()
        config.hidden_size = cfg.get('hidden_size', 768)
        config.num_layers = cfg.get('num_layers', 12)
        config.num_heads = cfg.get('num_heads', 12)
        config.seq_length = cfg.get('seq_length', 512)
    else:
        config = SimpleGPTConfig()

    print(f"Model config: {config.num_layers} layers, {config.hidden_size} hidden, {config.num_heads} heads")

    model = SimpleGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device).to(torch.bfloat16)
    model.eval()

    return model, config


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=100, temperature=0.8, top_k=40, device='cuda'):
    """Generate text from a prompt."""

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    print(prompt, end="", flush=True)

    # Generate tokens
    for _ in range(max_tokens):
        # Crop to max sequence length
        if tokens.size(1) > model.config.seq_length:
            tokens = tokens[:, -model.config.seq_length:]

        # Forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, _ = model(tokens)

        # Get logits for last token
        logits = logits[:, -1, :] / temperature

        # Top-k sampling
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append and decode
        tokens = torch.cat([tokens, next_token], dim=1)

        # Print the new token
        new_text = tokenizer.decode([next_token.item()])
        print(new_text, end="", flush=True)

        # Stop at end of text token (50256 for GPT-2)
        if next_token.item() == 50256:
            break

    print("\n" + "-" * 50)

    # Return full generated text
    return tokenizer.decode(tokens[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained GPT model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/simple-gpt/model_final.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="The meaning of life is",
                        help="Text prompt to continue")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0 = disabled)")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Load model
    model, config = load_model(args.checkpoint, device)

    # Generate
    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )


if __name__ == "__main__":
    main()
