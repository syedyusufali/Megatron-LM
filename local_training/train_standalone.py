#!/usr/bin/env python3
"""
Standalone GPT Training Script for Megatron-LM
==============================================
This script provides a simplified, self-contained training loop using Megatron-Core.
Perfect for understanding how Megatron-LM works and for customization.

Usage:
    torchrun --standalone --nproc_per_node=1 local_training/train_standalone.py

This trains a small GPT model (~25M params) using mock data for demonstration.
Modify the configuration section to customize the model and training.
"""

import os
import sys
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Iterator
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from megatron.core import parallel_state
from megatron.core import dist_checkpointing
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.datasets.utils import compile_helpers
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.training.tokenizer.tokenizer import _NullTokenizer


# ============================================================================
# CONFIGURATION - Modify these values to customize training
# ============================================================================

class TrainingConfig:
    """Training configuration for GPT model."""

    # Model architecture
    num_layers: int = 6              # Number of transformer layers
    hidden_size: int = 512           # Hidden dimension size
    num_attention_heads: int = 8     # Number of attention heads
    ffn_hidden_size: int = 2048      # Feed-forward network hidden size (4 * hidden_size)
    vocab_size: int = 50257          # GPT-2 vocabulary size
    max_sequence_length: int = 512   # Maximum sequence length

    # Training hyperparameters
    micro_batch_size: int = 8        # Batch size per GPU
    learning_rate: float = 1e-4      # Learning rate
    weight_decay: float = 0.1        # Weight decay for AdamW
    max_iterations: int = 1000       # Total training iterations
    log_interval: int = 10           # Log every N iterations
    save_interval: int = 500         # Save checkpoint every N iterations

    # Paths
    checkpoint_dir: str = "checkpoints/standalone"

    # Device settings
    use_bf16: bool = True            # Use bfloat16 mixed precision
    seed: int = 42                   # Random seed


config = TrainingConfig()


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def initialize_distributed() -> None:
    """Initialize torch.distributed and Megatron-Core for single GPU training."""
    parallel_state.destroy_model_parallel()

    # Get distributed settings from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.cuda.set_device(local_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
    else:
        # Single GPU - initialize with gloo for compatibility
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.distributed.init_process_group(
            backend="gloo",
            rank=0,
            world_size=1
        )

    # Initialize Megatron model parallel (no parallelism for single GPU)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1
    )


def create_model() -> GPTModel:
    """Create and return a GPT model based on configuration."""

    transformer_config = TransformerConfig(
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        ffn_hidden_size=config.ffn_hidden_size,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16 if config.use_bf16 else torch.float32,
        bf16=config.use_bf16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )

    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=config.vocab_size,
        max_sequence_length=config.max_sequence_length,
    )

    # Count parameters
    num_params = sum(p.numel() for p in gpt_model.parameters())
    print(f"Model created with {num_params:,} parameters ({num_params/1e6:.1f}M)")

    return gpt_model


def get_data_iterator() -> Iterator:
    """Create a mock dataset and return a data iterator."""

    # Compile dataset helpers
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    dataset_config = GPTDatasetConfig(
        random_seed=config.seed,
        sequence_length=config.max_sequence_length,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=_NullTokenizer(vocab_size=config.vocab_size),
        mid_level_dataset_surplus=0.005,
    )

    # Create mock dataset for demonstration
    # In production, use actual tokenized data instead
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset,
        [10000, None, None],  # train samples, valid samples, test samples
        lambda: True,
        dataset_config
    ).build()

    train_dataloader = DataLoader(
        datasets[0],
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return iter(train_dataloader)


def forward_step(
    data_iterator: Iterator,
    model: torch.nn.Module,
    device: torch.device
) -> Tuple[torch.Tensor, Callable]:
    """Forward step function for training."""

    def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm_loss": loss}

    # Get batch from iterator
    data = next(data_iterator)
    tokens = data["tokens"].to(device)
    attention_mask = data["attention_mask"].to(device)
    position_ids = data["position_ids"].to(device)
    labels = data["labels"].to(device)
    loss_mask = data["loss_mask"].to(device)

    # Forward pass
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                   iteration: int, checkpoint_dir: str) -> None:
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"iter_{iteration:07d}")

    # Get underlying model if wrapped
    model_to_save = model.module if hasattr(model, "module") else model

    torch.save({
        "iteration": iteration,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": vars(config)
    }, f"{checkpoint_path}.pt")

    print(f"Saved checkpoint to {checkpoint_path}.pt")


def train() -> None:
    """Main training loop."""

    print("=" * 60)
    print("Megatron-LM Standalone Training")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Layers: {config.num_layers}")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  Attention Heads: {config.num_attention_heads}")
    print(f"  Sequence Length: {config.max_sequence_length}")
    print(f"  Batch Size: {config.micro_batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Max Iterations: {config.max_iterations}")
    print(f"  Using BF16: {config.use_bf16}")
    print("=" * 60)

    # Initialize distributed training
    print("\nInitializing distributed training...")
    initialize_distributed()
    model_parallel_cuda_manual_seed(config.seed)

    # Create model
    print("\nCreating model...")
    device = torch.device("cuda")
    model = create_model()
    model.to(device)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )

    # Get data iterator
    print("\nPreparing data...")
    data_iterator = get_data_iterator()

    # Get forward/backward function
    forward_backward_func = get_forward_backward_func()

    # Training loop
    print("\nStarting training...")
    print("-" * 60)

    total_loss = 0.0
    for iteration in range(1, config.max_iterations + 1):
        optimizer.zero_grad()

        # Forward and backward pass
        try:
            losses_reduced = forward_backward_func(
                forward_step_func=lambda data_iter, model: forward_step(data_iter, model, device),
                data_iterator=data_iterator,
                model=model,
                num_microbatches=1,
                seq_length=config.max_sequence_length,
                micro_batch_size=config.micro_batch_size,
                decoder_seq_length=config.max_sequence_length,
                forward_only=False,
            )
        except StopIteration:
            # Reinitialize iterator if exhausted
            data_iterator = get_data_iterator()
            continue

        # Optimizer step
        optimizer.step()

        # Logging
        if losses_reduced and len(losses_reduced) > 0:
            loss = losses_reduced[0].get("lm_loss", torch.tensor(0.0))
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            total_loss += loss

            if iteration % config.log_interval == 0:
                avg_loss = total_loss / config.log_interval
                print(f"Iteration {iteration:6d} | Loss: {avg_loss:.4f}")
                total_loss = 0.0

        # Save checkpoint
        if iteration % config.save_interval == 0:
            save_checkpoint(model, optimizer, iteration, config.checkpoint_dir)

    print("-" * 60)
    print("Training complete!")

    # Final save
    save_checkpoint(model, optimizer, config.max_iterations, config.checkpoint_dir)

    # Cleanup
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train()
