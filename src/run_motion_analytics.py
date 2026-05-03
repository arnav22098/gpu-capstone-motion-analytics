from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU-accelerated motion analytics on a synthetic video dataset.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing frames.npy and input metadata.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where output CSVs and videos are written.")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory where proof artifacts are written.")
    parser.add_argument("--presentation-dir", type=Path, required=True, help="Directory for generated presentation support files.")
    parser.add_argument("--batch-size", type=int, default=64, help="Number of frames per GPU batch.")
    parser.add_argument("--cpu-subset", type=int, default=180, help="Number of frames used for CPU baseline timing.")
    return parser.parse_args()


def gaussian_kernel(device: torch.device) -> torch.Tensor:
    kernel = torch.tensor(
        [
            [1.0, 4.0, 6.0, 4.0, 1.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [6.0, 24.0, 36.0, 24.0, 6.0],
            [4.0, 16.0, 24.0, 16.0, 4.0],
            [1.0, 4.0, 6.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, 5, 5)


def load_frames(input_dir: Path) -> np.ndarray:
    frame_path = input_dir / "frames.npy"
    if not frame_path.exists():
        raise FileNotFoundError(f"Expected {frame_path} to exist.")
    return np.load(frame_path)


def run_gpu_pipeline(frames: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This project requires GPU execution.")

    device = torch.device("cuda")
    kernel = gaussian_kernel(device)

    transfer_to_gpu_s = 0.0
    compute_s = 0.0
    transfer_to_cpu_s = 0.0

    motion_scores_batches: list[np.ndarray] = []
    masks_batches: list[np.ndarray] = []

    previous_gray: torch.Tensor | None = None

    for start in range(0, frames.shape[0], batch_size):
        batch = frames[start : start + batch_size]

        t0 = time.perf_counter()
        batch_tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        transfer_to_gpu_s += time.perf_counter() - t0

        t1 = time.perf_counter()
        gray = 0.299 * batch_tensor[:, 0:1] + 0.587 * batch_tensor[:, 1:2] + 0.114 * batch_tensor[:, 2:3]
        blurred = F.conv2d(gray, kernel, padding=2)
        previous_shifted = torch.cat([previous_gray, blurred[:-1]], dim=0) if previous_gray is not None else torch.cat([blurred[:1], blurred[:-1]], dim=0)
        diff = torch.abs(blurred - previous_shifted)
        motion_mask = (diff > 18.0).float()
        motion_scores = motion_mask.mean(dim=(1, 2, 3))
        torch.cuda.synchronize()
        compute_s += time.perf_counter() - t1

        t2 = time.perf_counter()
        motion_scores_batches.append(motion_scores.detach().cpu().numpy())
        masks_batches.append((motion_mask.squeeze(1) * 255.0).to(dtype=torch.uint8).cpu().numpy())
        torch.cuda.synchronize()
        transfer_to_cpu_s += time.perf_counter() - t2

        previous_gray = blurred[-1:].detach()

    motion_scores_full = np.concatenate(motion_scores_batches, axis=0)
    masks_full = np.concatenate(masks_batches, axis=0)

    timings = {
        "gpu_name": torch.cuda.get_device_name(0),
        "host_to_device_s": transfer_to_gpu_s,
        "gpu_compute_s": compute_s,
        "device_to_host_s": transfer_to_cpu_s,
        "total_gpu_pipeline_s": transfer_to_gpu_s + compute_s + transfer_to_cpu_s,
    }
    return motion_scores_full, masks_full, timings


def run_cpu_baseline(frames: np.ndarray) -> float:
    device = torch.device("cpu")
    kernel = gaussian_kernel(device)

    start = time.perf_counter()
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
    gray = 0.299 * tensor[:, 0:1] + 0.587 * tensor[:, 1:2] + 0.114 * tensor[:, 2:3]
    blurred = F.conv2d(gray, kernel, padding=2)
    shifted = torch.cat([blurred[:1], blurred[:-1]], dim=0)
    diff = torch.abs(blurred - shifted)
    _ = (diff > 18.0).float().mean(dim=(1, 2, 3)).numpy()
    return time.perf_counter() - start


def detect_events(motion_scores: np.ndarray) -> list[dict[str, float | int]]:
    rolling = np.convolve(motion_scores, np.ones(11) / 11.0, mode="same")
    threshold = float(max(rolling.mean() + 1.1 * rolling.std(), np.percentile(rolling, 84)))
    active = rolling >= threshold

    raw_events: list[tuple[int, int]] = []
    start_idx: int | None = None
    min_duration = 8

    for idx, is_active in enumerate(active):
        if is_active and start_idx is None:
            start_idx = idx
        elif not is_active and start_idx is not None:
            if idx - start_idx >= min_duration:
                raw_events.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None and len(active) - start_idx >= min_duration:
        raw_events.append((start_idx, len(active) - 1))

    merged_events: list[tuple[int, int]] = []
    for start_frame, end_frame in raw_events:
        if merged_events and start_frame - merged_events[-1][1] <= 8:
            merged_events[-1] = (merged_events[-1][0], end_frame)
        else:
            merged_events.append((start_frame, end_frame))

    events = []
    for start_frame, end_frame in merged_events:
        peak = float(rolling[start_frame : end_frame + 1].max())
        events.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_frames": end_frame - start_frame + 1,
                "peak_motion_score": peak,
            }
        )

    return events


def save_csvs(output_dir: Path, motion_scores: np.ndarray, events: list[dict[str, float | int]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "motion_scores.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "motion_score"])
        for frame_index, score in enumerate(motion_scores):
            writer.writerow([frame_index, f"{float(score):.6f}"])

    with (output_dir / "detected_events.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["start_frame", "end_frame", "duration_frames", "peak_motion_score"])
        writer.writeheader()
        for event in events:
            row = dict(event)
            row["peak_motion_score"] = f"{float(row['peak_motion_score']):.6f}"
            writer.writerow(row)


def save_heatmap(output_dir: Path, masks: np.ndarray) -> Path:
    heatmap = masks.mean(axis=0)
    normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    heatmap_path = output_dir / "motion_heatmap.png"
    cv2.imwrite(str(heatmap_path), colored)
    return heatmap_path


def save_annotated_video(output_dir: Path, frames: np.ndarray, motion_scores: np.ndarray, events: list[dict[str, float | int]]) -> Path:
    output_path = output_dir / "annotated_motion_video.mp4"
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for annotated output.")

    event_ranges = [(int(event["start_frame"]), int(event["end_frame"])) for event in events]

    for frame_index, frame in enumerate(frames):
        annotated = frame.copy()
        cv2.putText(annotated, f"Frame {frame_index}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated, f"Motion score {motion_scores[frame_index]:.4f}", (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        is_event = any(start <= frame_index <= end for start, end in event_ranges)
        if is_event:
            cv2.putText(annotated, "Detected event", (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 240, 255), 2)
        writer.write(annotated)

    writer.release()
    return output_path


def save_contact_sheet(artifacts_dir: Path, frames: np.ndarray, masks: np.ndarray, motion_scores: np.ndarray, events: list[dict[str, float | int]]) -> Path:
    indices = [0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1, max(0, len(frames) // 2 - 30)]
    indices = list(dict.fromkeys(indices))[:6]

    tile_w = frames.shape[2]
    tile_h = frames.shape[1]
    canvas = Image.new("RGB", (tile_w * 2, tile_h * len(indices)), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)

    event_ranges = [(int(event["start_frame"]), int(event["end_frame"])) for event in events]

    for row, index in enumerate(indices):
        input_image = Image.fromarray(cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB))
        mask_image = Image.fromarray(masks[index], mode="L").convert("RGB")
        mask_image = ImageOps.autocontrast(mask_image)

        top = row * tile_h
        canvas.paste(input_image, (0, top))
        canvas.paste(mask_image, (tile_w, top))

        label = f"frame={index} score={motion_scores[index]:.4f}"
        if any(start <= index <= end for start, end in event_ranges):
            label += " event"
        draw.text((8, top + 8), label, fill=(255, 255, 255))

    output_path = artifacts_dir / "sample_motion_results.png"
    canvas.save(output_path)
    return output_path


def write_summary(
    artifacts_dir: Path,
    frame_count: int,
    width: int,
    height: int,
    gpu_timings: dict[str, float],
    cpu_baseline_s: float,
    event_count: int,
) -> None:
    summary = {
        "frame_count": frame_count,
        "resolution": {"width": width, "height": height},
        "gpu_timings": gpu_timings,
        "cpu_subset_seconds": cpu_baseline_s,
        "event_count": event_count,
    }
    (artifacts_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "GPU Specialization Capstone execution log",
        f"Frames processed: {frame_count}",
        f"Resolution: {width}x{height}",
        f"GPU: {gpu_timings['gpu_name']}",
        f"Host to device copy: {gpu_timings['host_to_device_s']:.6f} s",
        f"GPU compute: {gpu_timings['gpu_compute_s']:.6f} s",
        f"Device to host copy: {gpu_timings['device_to_host_s']:.6f} s",
        f"Total GPU pipeline: {gpu_timings['total_gpu_pipeline_s']:.6f} s",
        f"CPU subset baseline: {cpu_baseline_s:.6f} s",
        f"Detected events: {event_count}",
    ]
    (artifacts_dir / "execution_log.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_presentation_notes(presentation_dir: Path, event_count: int, gpu_timings: dict[str, float]) -> None:
    presentation_dir.mkdir(parents=True, exist_ok=True)
    notes = f"""# Capstone Demo Outline

## Slide 1 - Project motivation
- This project demonstrates GPU-accelerated motion analytics on a synthetic traffic video.
- The goal is to show a realistic computer-vision style workload that benefits from CUDA-backed batch processing.

## Slide 2 - Input dataset
- The dataset contains 720 generated frames at 256x256 resolution.
- The synthetic scene includes normal traffic, a rush-hour burst, and a pedestrian crossing event.

## Slide 3 - GPU pipeline
- Frames are moved to the GPU in batches.
- The pipeline performs grayscale conversion, Gaussian blur, temporal differencing, and threshold-based motion masking.
- Motion scores and heatmaps are aggregated from GPU results.

## Slide 4 - Outputs
- CSV output files track frame-level motion scores and detected events.
- Visual outputs include a motion heatmap, annotated output video, and side-by-side frame summaries.

## Slide 5 - Results
- Total GPU pipeline time: {gpu_timings['total_gpu_pipeline_s']:.4f} seconds.
- Detected events in the run: {event_count}.
- The project shows how GPU acceleration can support large video-style analytics workloads.

## Slide 6 - Lessons learned
- Batched tensor operations make it practical to process long frame sequences efficiently.
- GPU-friendly pipelines are a strong fit for video analytics, surveillance, and robotics-style workloads.
"""
    (presentation_dir / "DEMO_OUTLINE.md").write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    args.presentation_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(args.input_dir)
    motion_scores, masks, gpu_timings = run_gpu_pipeline(frames, args.batch_size)
    cpu_baseline_s = run_cpu_baseline(frames[: min(args.cpu_subset, len(frames))])
    events = detect_events(motion_scores)

    save_csvs(args.output_dir, motion_scores, events)
    save_heatmap(args.output_dir, masks)
    save_annotated_video(args.output_dir, frames, motion_scores, events)
    save_contact_sheet(args.artifacts_dir, frames, masks, motion_scores, events)
    write_summary(args.artifacts_dir, len(frames), frames.shape[2], frames.shape[1], gpu_timings, cpu_baseline_s, len(events))
    write_presentation_notes(args.presentation_dir, len(events), gpu_timings)

    print(f"Processed {len(frames)} frames on the GPU.")
    print(f"Detected {len(events)} motion events.")
    print(f"Outputs written to {args.output_dir}")
    print(f"Artifacts written to {args.artifacts_dir}")


if __name__ == "__main__":
    main()
