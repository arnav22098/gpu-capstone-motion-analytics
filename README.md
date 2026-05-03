# GPU Motion Analytics Capstone Project

This repository contains a GPU-accelerated capstone project for the Coursera GPU Specialization. The project analyzes a synthetic traffic-style video stream using CUDA-backed PyTorch tensor operations. It demonstrates how batched GPU computation can support motion analysis, event detection, CSV reporting, and visual analytics outputs.

The repository layout follows the structure of the course project template:

- clear `README.md`
- command line execution
- build and run support files
- input and output data
- proof-of-execution artifacts
- presentation support materials

## Project overview

The capstone simulates a simple traffic monitoring scenario. A synthetic video generator creates 720 frames containing moving vehicle-like shapes and two event windows:

- a rush-hour traffic burst
- a pedestrian crossing sequence

The GPU analytics pipeline processes the full frame sequence in batches and performs:

1. RGB to grayscale conversion on the GPU
2. Gaussian smoothing on the GPU
3. Frame-to-frame temporal differencing on the GPU
4. Threshold-based motion mask generation on the GPU
5. Motion score aggregation and event detection

## Why this is a strong capstone project

- It uses real GPU computation on a non-trivial frame sequence.
- It produces multiple output types: images, video, CSV files, logs, and presentation notes.
- It is more substantial than a single image transform because it combines data generation, temporal analytics, event detection, and reporting.
- It maps cleanly to practical applications such as surveillance analytics, robotics perception, and traffic monitoring.

## Repository structure

- `src/generate_synthetic_video.py`: creates the synthetic input video and frame data
- `src/run_motion_analytics.py`: runs the GPU analytics pipeline
- `Makefile`: Linux-style runner aligned with the course template
- `run.sh`: Linux / Coursera lab execution entry point
- `run.ps1`: Windows execution entry point
- `data/input/`: generated video, sampled input frames, and frame tensor data
- `data/output/`: generated CSVs, heatmap image, and annotated video
- `artifacts/`: execution logs and visual proof artifacts
- `presentation/`: demo and presentation notes

## Requirements

Install dependencies first:

```bash
pip install -r requirements.txt
```

Required environment:

- Python 3.10+
- NVIDIA GPU
- CUDA-capable PyTorch installation

## How to run

### Linux / Coursera lab

```bash
pip install -r requirements.txt
chmod +x run.sh
./run.sh
```

Or:

```bash
make run
```

### Windows PowerShell

```powershell
pip install -r requirements.txt
./run.ps1
```

### Manual commands

```bash
python src/generate_synthetic_video.py --output-dir data/input --frames 720 --width 256 --height 256 --seed 22098
python src/run_motion_analytics.py --input-dir data/input --output-dir data/output --artifacts-dir artifacts --presentation-dir presentation --batch-size 64 --cpu-subset 180
```

## Command line arguments

Generate a different synthetic dataset:

```bash
python src/generate_synthetic_video.py --output-dir data/input --frames 900 --width 320 --height 240 --seed 1234
```

Run analytics with a different batch size:

```bash
python src/run_motion_analytics.py --input-dir data/input --output-dir data/output --artifacts-dir artifacts --presentation-dir presentation --batch-size 32 --cpu-subset 120
```

## Main outputs

After a successful run, the project generates:

- `data/output/motion_scores.csv`
- `data/output/detected_events.csv`
- `data/output/motion_heatmap.png`
- `data/output/annotated_motion_video.mp4`
- `artifacts/execution_log.txt`
- `artifacts/benchmark_summary.json`
- `artifacts/sample_motion_results.png`
- `presentation/DEMO_OUTLINE.md`

## Submission tips

For the Coursera submission, you can provide:

1. the public repository URL
2. a zip file containing the artifact images, logs, and selected CSV outputs
3. a short project description based on the repository contents
4. a project demo or recorded walkthrough using the notes in `presentation/DEMO_OUTLINE.md`
