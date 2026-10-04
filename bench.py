#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import onnxruntime as ort
import psutil
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

def get_process_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def benchmark_onnx(model_path: str, runs: int = 100, warmup: int = 10):
    if not os.path.exists(model_path):
        console.print(f"[bold red]Ошибка:[/bold red] Файл '[bold yellow]{model_path}[/bold yellow]' не найден.")
        return

    console.print(
        Panel.fit(
            f"[bold green]ONNX Model Benchmark[/bold green]\n"
            f"Model: [cyan]{os.path.basename(model_path)}[/cyan]\n"
            f"Runs: [yellow]{runs}[/yellow] | Warmup: [yellow]{warmup}[/yellow]",
            title="onnx-bench-cli",
            border_style="green"
        )
    )

    mem_before = get_process_memory_mb()
    start_load = time.perf_counter()
    
    try:
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        console.print(f"[bold red]Ошибка загрузки ONNX модели:[/bold red] {e}")
        return

    load_time_ms = (time.perf_counter() - start_load) * 1000

    input_data = {}
    for input_meta in session.get_inputs():
        shape = [s if isinstance(s, int) and s > 0 else 1 for s in input_meta.shape]
        
        dtype = np.float32
        if "int64" in input_meta.type:
            dtype = np.int64
        elif "int32" in input_meta.type:
            dtype = np.int32

        input_data[input_meta.name] = np.random.randn(*shape).astype(dtype)

    # Warmup
    for _ in range(warmup):
        session.run(None, input_data)

    # Main Benchmark Loop
    latencies = []
    mem_peaks = []

    for _ in track(range(runs), description="[cyan]Running inference...[/cyan]"):
        t0 = time.perf_counter()
        session.run(None, input_data)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
        mem_peaks.append(get_process_memory_mb())

    # Metrics calculation
    latencies = np.array(latencies)
    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p95_lat = np.percentile(latencies, 95)
    std_lat = np.std(latencies)
    fps = 1000.0 / avg_lat if avg_lat > 0 else 0
    ram_used_mb = max(mem_peaks) - mem_before

    # Summary Table
    table = Table(title="📊 Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Model Load Time", f"{load_time_ms:.2f} ms")
    table.add_row("Mean Latency", f"[bold green]{avg_lat:.2f} ms[/bold green]")
    table.add_row("Min / Max Latency", f"{min_lat:.2f} ms / {max_lat:.2f} ms")
    table.add_row("P95 Latency", f"{p95_lat:.2f} ms")
    table.add_row("Std Dev", f"±{std_lat:.2f} ms")
    table.add_row("Throughput", f"[bold cyan]{fps:.1f} FPS[/bold cyan]")
    table.add_row("Peak RAM Usage", f"{max(0.0, ram_used_mb):.2f} MB")

    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="CLI tool to benchmark ONNX models.")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to .onnx model file")
    parser.add_argument("-r", "--runs", type=int, default=100, help="Number of inference runs (default: 100)")
    parser.add_argument("-w", "--warmup", type=int, default=10, help="Number of warmup runs (default: 10)")

    args = parser.parse_args()
    benchmark_onnx(args.model, args.runs, args.warmup)

if __name__ == "__main__":
    main()