#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

pythonBin=.venv/bin/python
hfCli=.venv/bin/huggingface-cli

echo "== base model -> train/base/"
"$hfCli" download Qwen/Qwen2.5-0.5B-Instruct --local-dir train/base

echo "== extract law articles from the QA dataset"
"$pythonBin" data/extract_laws.py

echo "== build the law index"
"$pythonBin" data/build_index.py

echo "== make the train / eval / test sets"
"$pythonBin" data/prepare.py --explode generated_qa_pairs --q-field question --a-field answer \
  --gold-field article_content --law-field doc_name
