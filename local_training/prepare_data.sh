#!/bin/bash
# Data preparation script for Megatron-LM training
# Downloads and preprocesses text data for GPT training

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR/data"
VOCAB_DIR="$DATA_DIR/vocab"
PROCESSED_DIR="$DATA_DIR/processed"

echo "=== Megatron-LM Data Preparation ==="
echo ""
echo "Data directory: $DATA_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR/raw"
mkdir -p "$VOCAB_DIR"
mkdir -p "$PROCESSED_DIR"

# Download GPT-2 tokenizer vocab and merges
echo "Downloading GPT-2 tokenizer files..."
if [ ! -f "$VOCAB_DIR/gpt2-vocab.json" ]; then
    wget -O "$VOCAB_DIR/gpt2-vocab.json" \
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json"
fi

if [ ! -f "$VOCAB_DIR/gpt2-merges.txt" ]; then
    wget -O "$VOCAB_DIR/gpt2-merges.txt" \
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt"
fi

echo ""
echo "Downloading training data (WikiText-103)..."

# Download WikiText-103 dataset
cd "$DATA_DIR/raw"
if [ ! -f "wikitext-103-raw-v1.zip" ]; then
    wget "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    unzip -o wikitext-103-raw-v1.zip
fi

# Combine wiki files into single training file
echo ""
echo "Preparing text files..."
cat wikitext-103-raw/wiki.train.raw > train.txt
cat wikitext-103-raw/wiki.valid.raw > valid.txt
cat wikitext-103-raw/wiki.test.raw > test.txt

cd "$ROOT_DIR"

echo ""
echo "Preprocessing data for Megatron-LM..."
echo "This may take several minutes depending on your CPU..."
echo ""

# Preprocess training data
python tools/preprocess_data.py \
    --input "$DATA_DIR/raw/train.txt" \
    --output-prefix "$PROCESSED_DIR/wikitext103_train" \
    --vocab-file "$VOCAB_DIR/gpt2-vocab.json" \
    --merge-file "$VOCAB_DIR/gpt2-merges.txt" \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 4

# Preprocess validation data
python tools/preprocess_data.py \
    --input "$DATA_DIR/raw/valid.txt" \
    --output-prefix "$PROCESSED_DIR/wikitext103_valid" \
    --vocab-file "$VOCAB_DIR/gpt2-vocab.json" \
    --merge-file "$VOCAB_DIR/gpt2-merges.txt" \
    --tokenizer-type GPT2BPETokenizer \
    --append-eod \
    --workers 4

echo ""
echo "=== Data Preparation Complete! ==="
echo ""
echo "Processed files:"
ls -lh "$PROCESSED_DIR/"
echo ""
echo "You can now run training with: ./local_training/train.sh"
