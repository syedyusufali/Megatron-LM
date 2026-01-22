#!/bin/bash
# Quick training script with smaller model for testing
# Trains a ~25M parameter model - useful for quick debugging and verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Smaller model for quick testing (~25M parameters)
NUM_LAYERS=6
HIDDEN_SIZE=512
NUM_ATTENTION_HEADS=8
SEQ_LENGTH=512

# Training settings
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=64
LR=1e-3
TRAIN_ITERS=10000
SAVE_INTERVAL=1000
EVAL_INTERVAL=500
EVAL_ITERS=50

# Paths
DATA_PATH="$ROOT_DIR/data/processed/wikitext103_train_text_document"
VALID_DATA_PATH="$ROOT_DIR/data/processed/wikitext103_valid_text_document"
VOCAB_FILE="$ROOT_DIR/data/vocab/gpt2-vocab.json"
MERGE_FILE="$ROOT_DIR/data/vocab/gpt2-merges.txt"
CHECKPOINT_DIR="$ROOT_DIR/checkpoints/gpt-25m-test"
TENSORBOARD_DIR="$ROOT_DIR/tensorboard/gpt-25m-test"

mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$TENSORBOARD_DIR"

if [ ! -f "${DATA_PATH}.bin" ]; then
    echo "Error: Training data not found. Run prepare_data.sh first."
    exit 1
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo "=== Quick Test Training (25M model) ==="
echo ""

torchrun --standalone --nproc_per_node=1 "$ROOT_DIR/pretrain_gpt.py" \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr $LR \
    --min-lr 1e-4 \
    --lr-warmup-iters 500 \
    --lr-decay-iters $TRAIN_ITERS \
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
    --bf16 \
    --use-flash-attn \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --transformer-impl local \
    2>&1 | tee "$ROOT_DIR/training_log_small.txt"

echo "Test training complete!"
