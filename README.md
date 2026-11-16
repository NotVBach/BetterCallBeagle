## Project: Better call Beagle

1. Objective:
Build an assistant that answers Vietnamese legal questions entirely on a BeagleBone Black Rev C using **llama.cpp**

Target hardware:
- BeagleBone Black Rev C
- CPU: ARM Cortex-A8, 1 GHz (very old cpu)
- RAM: 512 MB DDR3
- Architecture: 32-bit ARMv7
- SIMD: NEON
- Storage: 4 GB eMMC
- OS: Debian Linux

2. Technical
- Model: Qwen2.5-0.5B-Instruct
- LoRA fine-tuned, Q4_K_M GGUF (~400 MB) and SQLite FTS5 retrieval over law articles.

3. Run from scratch
- Requirements:
```bash
sudo apt install build-essential cmake git python3-venv \
     gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf binutils-arm-linux-gnueabihf
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
huggingface-cli download --repo-type dataset thangvip/vietnamese-legal-qa --local-dir data/raw/qa
```

- Data processing: `data/process.sh`

- Fine-tuning: `python train/train_lora.py`

- Building and quantizing: `build/build.sh`

- Benchmarking: `python bench/bench.py`

- Run app: `python app.py`

4. Benchmark

`bench/results/laptop_default_20261116-152411`: laptop chip (x86_64, 12 threads), `beagle-q4_k_m.gguf`, default config (RAG, speculative decoding and cache on), 20 test questions.

| Metric (max) | AM3358 | Intel Core i5-13420H |
|---|---|---|
| Retrieval (ms) | | 151.2 |
| Time to first token (s) | | 1.90 |
| Whole answer (s) | | 9.70 |
| Prefill (tokens/s) | | 278.2 |
| Generation (tokens/s) | | 76.7 |
| llama-server memory (MB) | | 553.0 |
| Whole machine memory (MB) | | 6338.1 |
