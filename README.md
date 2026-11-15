## Project: Better call Beagle

1. OVERVIEW

Objective:
Build an assistant that answers Vietnamese legal questions **entirely on a BeagleBone Black Rev C** using **llama.cpp**

Target hardware:
- BeagleBone Black Rev C (2024)
- CPU: ARM Cortex-A8, 1 GHz
- RAM: 512 MB DDR3
- Architecture: 32-bit ARMv7
- SIMD: NEON
- Storage: 4 GB eMMC
- OS: Debian Linux

Model: Qwen2.5-0.5B-Instruct, LoRA fine-tuned, Q4_K_M GGUF (~400 MB), plus SQLite FTS5 retrieval over law articles.

Run app.py to enter chat mode
