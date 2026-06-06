#!/usr/bin/env bash
set -euo pipefail

COMPETITION="nycu-data-mining-assignment-3"
DATA_DIR="data"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI is not installed. Install it with: pip install kaggle"
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "Missing Kaggle API token at $HOME/.kaggle/kaggle.json"
  echo "Create one from Kaggle > Account > API > Create New Token."
  exit 1
fi

mkdir -p "$DATA_DIR"

kaggle competitions download \
  -c "$COMPETITION" \
  -p "$DATA_DIR"

ZIP_PATH="$DATA_DIR/$COMPETITION.zip"
if [ -f "$ZIP_PATH" ]; then
  unzip -o "$ZIP_PATH" -d "$DATA_DIR"
fi

echo "Data is ready in $DATA_DIR/"
