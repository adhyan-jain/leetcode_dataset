#!/bin/bash
set -e

echo "Starting Pass 1 (Qwen 7b) for missing rows up to 1000..."
PYTHONPATH=. .venv/bin/python scripts/pass1_enrich.py \
    --input "leetcode_dataset - lc.csv" \
    --output "data/intermediate/pass1_enriched.jsonl" \
    --failures "data/failures/pass1_failures.jsonl" \
    --model "qwen2.5-coder:7b" \
    --limit 1000 \
    --resume

echo "Pass 1 Complete. Starting Pass 2 (Taxonomy Build)..."
PYTHONPATH=. .venv/bin/python scripts/build_taxonomy.py \
    --input "data/intermediate/pass1_enriched.jsonl" \
    --taxonomy-output "data/taxonomy/taxonomy.json" \
    --mapping-output "data/taxonomy/mapping.json" \
    --report-output "data/taxonomy/report.md"

echo "Pass 2 Complete. Starting Pass 3 (Constrained Enrich)..."
PYTHONPATH=. .venv/bin/python scripts/pass3_constrained_enrich.py \
    --input "leetcode_dataset - lc.csv" \
    --taxonomy "data/taxonomy/taxonomy.json" \
    --label-mapping "data/taxonomy/mapping.json" \
    --output "data/final/final_enriched.jsonl" \
    --failures "data/failures/pass3_failures.jsonl" \
    --report "data/final/pass3_report.md" \
    --model "qwen2.5-coder:7b" \
    --limit 1000 \
    --resume

echo "Pipeline complete."
