#!/bin/bash
# Training script for GPT model on single RTX 5080 GPU
# Trains a ~125M parameter model from scratch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ============================================================================
# CONFIGURATION - Modify these settings as needed
# ============================================================================

# Model architecture (~125M parameters, similar to GPT-2 Small)
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
SEQ_LENGTH=1024

# Training hyperparameters
MICRO_BATCH_SIZE=4                 # Per-GPU batch size (reduce if OOM)
GLOBAL_BATCH_SIZE=32               # Total batch size (with gradient accumulation)
LR=6e-4                            # Learning rate
MIN_LR=6e-5                        # Minimum learning rate
LR_WARMUP_ITERS=2000               # Warmup iterations
LR_DECAY_ITERS=300000              # LR decay iterations
TRAIN_ITERS=100000                 # Total training iterations
SAVE_INTERVAL=5000                 # Save checkpoint every N iterations
EVAL_INTERVAL=1000                 # Evaluate every N iterations
EVAL_ITERS=100                     # Number of evaluation iterations

# Paths
DATA_PATH="$ROOT_DIR/data/processed/wikitext103_train_text_document"
VALID_DATA_PATH="$ROOT_DIR/data/processed/wikitext103_valid_text_document"
VOCAB_FILE="$ROOT_DIR/data/vocab/gpt2-vocab.json"
MERGE_FILE="$ROOT_DIR/data/vocab/gpt2-merges.txt"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints/gpt-125m"
TENSORBOARD_DIR="$ROOT_DIR/tensorboard/gpt-125m"

# ============================================================================
# END CONFIGURATION
# ============================================================================

# Create output directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"

# Check if data exists
if [ ! -f "${DATA_PATH}.bin" ]; then
    echo "Error: Training data not found at $DATA_PATH"
    echo "Please run prepare_data.sh first: ./local_training/prepare_data.sh"
    exit 1
fi

# Set environment variables for single GPU training
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

# Calculate gradient accumulation steps
GRADIENT_ACC_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))

echo "=== Megatron-LM GPT Training ==="
echo ""
echo "Model Configuration:"
echo "  - Layers: $NUM_LAYERS"
echo "  - Hidden Size: $HIDDEN_SIZE"
echo "  - Attention Heads: $NUM_ATTENTION_HEADS"
echo "  - Sequence Length: $SEQ_LENGTH"
echo "  - Approximate Parameters: ~125M"
echo ""
echo "Training Configuration:"
echo "  - Micro Batch Size: $MICRO_BATCH_SIZE"
echo "  - Global Batch Size: $GLOBAL_BATCH_SIZE"
echo "  - Gradient Accumulation Steps: $GRADIENT_ACC_STEPS"
echo "  - Learning Rate: $LR"
echo "  - Total Iterations: $TRAIN_ITERS"
echo ""
echo "Checkpoints will be saved to: $CHECKPOINT_DIR"
echo "TensorBoard logs: $TENSORBOARD_DIR"
echo ""

# Launch training with torchrun for single GPU
torchrun --standalone --nproc_per_node=1 "$ROOT_DIR/pretrain_gpt.py" \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LR \
    --min-lr $MIN_LR \
    --lr-warmup-iters $LR_WARMUP_ITERS \
    --lr-decay-iters $LR_DECAY_ITERS \
    --lr-decay-style cosine \
    --train-iters $TRAIN_ITERS \
    --save-interval $SAVE_INTERVAL \
    --eval-interval $EVAL_INTERVAL \
    --eval-iters $EVAL_ITERS \
    --data-path $DATA_PATH \
    --valid-data-path $VALID_DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --tokenizer-type GPT2BPETokenizer \
    --save $CHECKPOINT_DIR \
    --load $CHECKPOINT_DIR \
    --tensorboard-dir $TENSORBOARD_DIR \
    --log-interval 10 \
    --split 969,30,1 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --bf16 \
    --use-flash-attn \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-bias-gelu-fusion \
    --no-async-tensor-model-parallel-allreduce \
    --use-rotary-position-embeddings \
    --no-position-embedding \
    --untie-embeddings-and-output-weights \
    --normalization RMSNorm \
    --disable-bias-linear \
    --transformer-impl local \
    --recompute-activations \
    --recompute-granularity selective \
    --sequence-parallel \
    2>&1 | tee "$ROOT_DIR/training_log.txt"

echo ""
echo "Training complete! Checkpoints saved to: $CHECKPOINT_DIR"
