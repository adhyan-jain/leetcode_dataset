#!/bin/bash
set -e

echo "Starting Pass 1 (Qwen 7b) for missing rows up to 1000..."
PYTHONPATH=. .venv/bin/python scripts/pass1_enrich.py \
    --input "leetcode_dataset - lc.csv" \
    --output "data/intermediate/pass1_enriched.jsonl" \
    --failures "data/failures/pass1_failures.jsonl" \
    --model "qwen2.5-coder:7b" \
    --resume

echo "Pipeline complete."
