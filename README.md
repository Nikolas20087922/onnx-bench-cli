cat << 'EOF' > README.md

A lightweight, terminal-based CLI utility designed to benchmark ONNX models on LINUX/macOS. It automotically measures model loading time, inference latency ($P_{95}$, mean, min/max), throughput (FPS), and peak memory consumption.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.15+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Key Features 

1)Auto-generated dummy inputs: Automatically detects model input shapes and data types.
2)Comprehensive Metrics: Calculates mean latency, $P_{95}$ percentile, standard deviation, FPS, and peak RAM usage via `psutil`.
3)Rich Terminal UI: Clean progress bars and formatted tables powered by `rich`.
4)Docker-ready: Easy deployment and execution inside containers.

## Option 1: Native Python

'''bash

# 1. Clone repository and create virtual environment
git clone[https://github.com/Nikolas20087922/onnx-bench-cli.git](https://github.com/Nikolas20087922/onnx-bench-cli.git)
cd onnx_bench-cli
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run benchmark
python3 bench.py --model squeezenet.onnx --runs 100 --warmup 10
