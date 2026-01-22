#!/usr/bin/env python3
"""
Simple GPT Training Script - Single GPU, No Distributed
=========================================================
A minimal training script that works directly without torchrun/distributed.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import time
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleGPTConfig:
    vocab_size: int = 50304  # GPT-2 vocab padded to multiple of 64
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    seq_length: int = 512
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.fc2 = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SimpleGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.seq_length, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Count parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    def forward(self, idx, targets=None):
        B, T = idx.shape

        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss


class MegatronDataset(Dataset):
    """Load preprocessed Megatron binary data."""
    def __init__(self, data_path, seq_length):
        self.seq_length = seq_length

        # Load the binary data
        bin_path = data_path + ".bin"
        idx_path = data_path + ".idx"

        if os.path.exists(bin_path):
            print(f"Loading data from {bin_path}")
            # Memory-map the file for efficiency
            self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
            self.num_tokens = len(self.data)
            print(f"Loaded {self.num_tokens:,} tokens ({self.num_tokens/1e6:.1f}M)")
        else:
            print(f"Data file not found: {bin_path}")
            print("Using random data instead...")
            self.data = None
            self.num_tokens = 10_000_000  # 10M fake tokens

    def __len__(self):
        return (self.num_tokens - 1) // self.seq_length

    def __getitem__(self, idx):
        if self.data is not None:
            start = idx * self.seq_length
            end = start + self.seq_length + 1
            if end > len(self.data):
                start = 0
                end = self.seq_length + 1
            chunk = torch.from_numpy(self.data[start:end].astype(np.int64))
        else:
            # Random fallback
            chunk = torch.randint(0, 50257, (self.seq_length + 1,))

        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class RandomDataset(Dataset):
    """Random token dataset for demonstration."""
    def __init__(self, vocab_size, seq_length, num_samples):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_length + 1,))
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


def train():
    print("=" * 60)
    print("Simple GPT Training on WikiText-103")
    print("=" * 60)

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Config
    config = SimpleGPTConfig()
    batch_size = 8
    num_iters = 1000  # Longer training
    lr = 3e-4
    warmup_iters = 100

    print(f"\nConfig:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Seq Length: {config.seq_length}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Iterations: {num_iters}")
    print(f"  Learning Rate: {lr}")

    # Model
    print("\nCreating model...")
    model = SimpleGPT(config).to(device)
    model = model.to(torch.bfloat16)  # Use BF16

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))

    # Learning rate scheduler with warmup
    def get_lr(it):
        if it < warmup_iters:
            return lr * it / warmup_iters
        # Cosine decay
        decay_ratio = (it - warmup_iters) / (num_iters - warmup_iters)
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return lr * 0.1 + coeff * (lr - lr * 0.1)

    # Data - try to load real data, fallback to random
    data_path = "data/processed/wikitext103_train_text_document"
    print(f"\nLoading dataset...")
    dataset = MegatronDataset(data_path, config.seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Training
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)

    model.train()
    total_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()
    tokens_processed = 0

    data_iter = iter(dataloader)
    for i in range(1, num_iters + 1):
        # Get batch
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x, y = x.to(device), y.to(device)

        # Update learning rate
        current_lr = get_lr(i)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        tokens_processed += batch_size * config.seq_length

        # Logging
        if i % 50 == 0:
            avg_loss = total_loss / 50
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            # Perplexity
            ppl = np.exp(avg_loss) if avg_loss < 20 else float('inf')

            print(f"Iter {i:5d} | Loss: {avg_loss:.4f} | PPL: {ppl:8.2f} | LR: {current_lr:.2e} | {tokens_per_sec:,.0f} tok/s")

            if avg_loss < best_loss:
                best_loss = avg_loss

            total_loss = 0.0

        # Save checkpoint periodically
        if i % 500 == 0:
            os.makedirs("checkpoints/simple-gpt", exist_ok=True)
            torch.save({
                'iteration': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss if 'avg_loss' in dir() else 0,
                'config': vars(config),
            }, f"checkpoints/simple-gpt/model_iter{i}.pt")
            print(f"  >> Checkpoint saved: checkpoints/simple-gpt/model_iter{i}.pt")

    # Final stats
    total_time = time.time() - start_time
    total_tokens = num_iters * batch_size * config.seq_length

    print("-" * 60)
    print(f"Training complete!")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average throughput: {total_tokens / total_time:,.0f} tokens/sec")
    print(f"Best loss: {best_loss:.4f}")
    print("-" * 60)

    # Final save
    os.makedirs("checkpoints/simple-gpt", exist_ok=True)
    torch.save({
        'iteration': num_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'config': vars(config),
    }, "checkpoints/simple-gpt/model_final.pt")
    print(f"\nFinal checkpoint saved to checkpoints/simple-gpt/model_final.pt")


if __name__ == "__main__":
    train()
