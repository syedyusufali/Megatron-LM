# Megatron-LM Local Training on RTX 5080

Train a GPT language model from scratch on your local RTX 5080 GPU using NVIDIA's Megatron-LM framework.

## Overview

This setup provides scripts to train GPT models locally:

| Script | Model Size | Parameters | Use Case |
|--------|-----------|------------|----------|
| `train.sh` | GPT-125M | ~125M | Full training run |
| `train_small.sh` | GPT-25M | ~25M | Quick testing |
| `train_standalone.py` | GPT-25M | ~25M | Learning/customization |

## Quick Start

### 1. Setup Environment

```bash
# Make scripts executable
chmod +x local_training/*.sh

# Install dependencies (creates conda environment)
./local_training/setup_environment.sh

# Activate environment
conda activate megatron-lm
```

### 2. Prepare Training Data

```bash
# Downloads WikiText-103 and preprocesses it for Megatron-LM
./local_training/prepare_data.sh
```

This downloads:
- GPT-2 tokenizer vocabulary
- WikiText-103 dataset (~500MB of Wikipedia text)

### 3. Start Training

**Option A: Full training (125M model)**
```bash
./local_training/train.sh
```

**Option B: Quick test (25M model)**
```bash
./local_training/train_small.sh
```

**Option C: Standalone Python script (for learning/customization)**
```bash
torchrun --standalone --nproc_per_node=1 local_training/train_standalone.py
```

## Model Configurations

### GPT-125M (train.sh)
- 12 layers, 768 hidden size, 12 attention heads
- Sequence length: 1024
- Micro batch size: 4
- Memory usage: ~10-12GB VRAM

### GPT-25M (train_small.sh)
- 6 layers, 512 hidden size, 8 attention heads
- Sequence length: 512
- Micro batch size: 8
- Memory usage: ~4-6GB VRAM

## Training on Your RTX 5080

Your RTX 5080 (16GB VRAM) can comfortably train:
- **125M model** with batch size 4 and sequence length 1024
- **350M model** with batch size 2 and gradient checkpointing

### Optimizations Enabled
- **Flash Attention 2**: Faster, memory-efficient attention
- **BF16 Mixed Precision**: Better numerical stability than FP16
- **Gradient Checkpointing**: Trade compute for memory
- **RoPE Embeddings**: Modern rotary positional embeddings

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir tensorboard/
```
Then open http://localhost:6006 in your browser.

### Training Logs
Logs are saved to `training_log.txt` in the root directory.

## Customizing Training

### Adjust Batch Size (if OOM)
Edit `train.sh` and reduce `MICRO_BATCH_SIZE`:
```bash
MICRO_BATCH_SIZE=2  # Reduce from 4 to 2
```

### Change Model Size
Edit the model architecture variables:
```bash
NUM_LAYERS=24        # More layers = bigger model
HIDDEN_SIZE=1024     # Larger hidden = bigger model
NUM_ATTENTION_HEADS=16
```

### Use Your Own Data
1. Prepare your text data as a single `.txt` file
2. Run preprocessing:
```bash
python tools/preprocess_data.py \
    --input your_data.txt \
    --output-prefix data/processed/your_data \
    --vocab-file data/vocab/gpt2-vocab.json \
    --merge-file data/vocab/gpt2-merges.txt \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 4
```
3. Update `DATA_PATH` in `train.sh`

## Checkpoints

Checkpoints are saved to `checkpoints/gpt-125m/` by default.

### Resume Training
Training automatically resumes from the latest checkpoint in `CHECKPOINT_DIR`.

### Convert to HuggingFace Format
After training, convert your checkpoint:
```bash
python tools/checkpoint/convert_mcore_to_hf.py \
    --load-path checkpoints/gpt-125m/iter_XXXXX \
    --save-path hf_model/ \
    --model-type gpt2
```

## Project Structure

```
Megatron_LM/
├── local_training/
│   ├── setup_environment.sh   # Install dependencies
│   ├── prepare_data.sh        # Download & preprocess data
│   ├── train.sh               # Main training script (125M)
│   ├── train_small.sh         # Quick test script (25M)
│   ├── train_standalone.py    # Standalone Python trainer
│   └── README.md              # This file
├── data/
│   ├── raw/                   # Raw downloaded data
│   ├── vocab/                 # Tokenizer files
│   └── processed/             # Preprocessed binary data
├── checkpoints/               # Saved model checkpoints
├── tensorboard/               # Training metrics
└── pretrain_gpt.py           # Main Megatron training script
```

## Troubleshooting

### Out of Memory (OOM)
1. Reduce `MICRO_BATCH_SIZE` (try 2 or 1)
2. Reduce `SEQ_LENGTH` (try 512)
3. Enable more aggressive gradient checkpointing

### CUDA Version Mismatch
Ensure PyTorch CUDA version matches your driver:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
nvidia-smi
```

### Flash Attention Not Working
Install manually:
```bash
pip install flash-attn --no-build-isolation
```

### Data Not Found
Ensure `prepare_data.sh` completed successfully:
```bash
ls -la data/processed/
# Should show .bin and .idx files
```

## Performance Tips

1. **Use Flash Attention**: Already enabled by default
2. **Use BF16**: Better than FP16 for training stability
3. **Increase batch size**: Use largest batch that fits in memory
4. **Use gradient accumulation**: Increase effective batch size

## Expected Training Speed (RTX 5080)

| Model | Tokens/sec | Time for 1B tokens |
|-------|------------|-------------------|
| 25M | ~50,000 | ~5.5 hours |
| 125M | ~15,000 | ~18 hours |
| 350M | ~5,000 | ~55 hours |

## Resources

- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-Core API](https://docs.nvidia.com/megatron-core/)
- [Training LLMs Guide](https://huggingface.co/docs/transformers/llm_tutorial)
