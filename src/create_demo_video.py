from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import textwrap
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1280
HEIGHT = 720
FPS = 24
BG = (15, 18, 28)
TEXT = (242, 245, 250)
ACCENT = (85, 180, 255)
MUTED = (170, 182, 196)
CARD = (28, 34, 48)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a silent demo video for the GPU capstone project.")
    parser.add_argument("--project-root", type=Path, required=True, help="Project root directory.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    return parser.parse_args()


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/segoeuib.ttf",
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/calibrib.ttf",
            ]
        )
    candidates.extend(
        [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def add_wrapped_text(draw: ImageDraw.ImageDraw, text: str, xy: tuple[int, int], font: ImageFont.ImageFont, fill: tuple[int, int, int], width: int, line_spacing: int = 8) -> int:
    x, y = xy
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width))
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + line_spacing
    return y


def seconds_to_frames(seconds: float) -> int:
    return int(round(seconds * FPS))


def canvas() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, WIDTH, 90), fill=(10, 12, 20))
    draw.rectangle((0, HEIGHT - 64, WIDTH, HEIGHT), fill=(10, 12, 20))
    return image


def title_slide(summary: dict) -> list[np.ndarray]:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title_font = get_font(42, bold=True)
    body_font = get_font(26)
    small_font = get_font(20)

    draw.text((60, 120), "GPU-Accelerated Motion Analytics", font=title_font, fill=TEXT)
    draw.text((60, 175), "Synthetic traffic video capstone demo", font=body_font, fill=ACCENT)

    bullets = [
        f"Processed {summary['frame_count']} frames at {summary['resolution']['width']}x{summary['resolution']['height']}",
        f"GPU: {summary['gpu_timings']['gpu_name']}",
        f"Total GPU pipeline time: {summary['gpu_timings']['total_gpu_pipeline_s']:.4f} seconds",
        f"Detected motion events: {summary['event_count']}",
    ]
    y = 260
    for bullet in bullets:
        draw.ellipse((68, y + 11, 80, y + 23), fill=ACCENT)
        draw.text((100, y), bullet, font=body_font, fill=TEXT)
        y += 54

    draw.rounded_rectangle((60, 520, 1220, 640), radius=24, fill=CARD)
    add_wrapped_text(
        draw,
        "This video shows the generated input scenario, the GPU analytics pipeline, the event detections, and the final proof artifacts that can be submitted with the project.",
        (90, 548),
        small_font,
        MUTED,
        width=72,
        line_spacing=10,
    )
    return [np.array(image)] * seconds_to_frames(4.0)


def terminal_slide(commands: list[str], log_lines: list[str]) -> list[np.ndarray]:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    mono_font = get_font(22)
    caption_font = get_font(20)

    draw.text((60, 120), "Execution flow", font=title_font, fill=TEXT)
    draw.rounded_rectangle((60, 180, 1220, 620), radius=20, fill=(7, 10, 16))
    draw.text((84, 202), "PowerShell / terminal", font=caption_font, fill=ACCENT)

    y = 248
    for command in commands:
        draw.text((90, y), f"> {command}", font=mono_font, fill=(132, 225, 132))
        y += 42
    y += 14
    for line in log_lines:
        draw.text((90, y), line, font=mono_font, fill=TEXT)
        y += 36
    return [np.array(image)] * seconds_to_frames(4.5)


def fit_image(path: Path, max_size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(max_size)
    return image


def collage_slide(project_root: Path) -> list[np.ndarray]:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    small_font = get_font(22)
    draw.text((60, 120), "Generated inputs and analytics outputs", font=title_font, fill=TEXT)

    sample = fit_image(project_root / "artifacts" / "sample_motion_results.png", (760, 520))
    heatmap = fit_image(project_root / "data" / "output" / "motion_heatmap.png", (360, 260))
    input_frame = fit_image(project_root / "data" / "input" / "input_frame_0240.png", (360, 220))

    image.paste(sample, (60, 180))
    image.paste(heatmap, (840, 180))
    image.paste(input_frame, (840, 460))

    draw.text((840, 450), "Motion heatmap", font=small_font, fill=ACCENT)
    draw.text((840, 420), "Sample input frame", font=small_font, fill=ACCENT)
    draw.text((60, 660), "Left: sampled frames with motion masks. Right: heatmap and source frame.", font=small_font, fill=MUTED)
    return [np.array(image)] * seconds_to_frames(4.0)


def make_chart(project_root: Path, dest: Path) -> Path:
    csv_path = project_root / "data" / "output" / "motion_scores.csv"
    event_path = project_root / "data" / "output" / "detected_events.csv"

    scores = []
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            scores.append(float(row["motion_score"]))

    events = []
    with event_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            events.append((int(row["start_frame"]), int(row["end_frame"])))

    plt.figure(figsize=(12, 4.6), dpi=160)
    plt.plot(scores, color="#51a7ff", linewidth=1.6)
    for start, end in events:
        plt.axvspan(start, end, color="#ff9050", alpha=0.25)
    plt.title("Frame-level motion score with detected events")
    plt.xlabel("Frame index")
    plt.ylabel("Motion score")
    plt.grid(alpha=0.18)
    plt.tight_layout()
    plt.savefig(dest)
    plt.close()
    return dest


def chart_slide(project_root: Path, summary: dict, temp_chart: Path) -> list[np.ndarray]:
    make_chart(project_root, temp_chart)
    chart = fit_image(temp_chart, (1120, 460))
    image = canvas()
    draw = ImageDraw.Draw(image)
    title_font = get_font(34, bold=True)
    body_font = get_font(22)
    draw.text((60, 120), "Detected events and quantitative results", font=title_font, fill=TEXT)
    image.paste(chart, (80, 180))

    draw.text((80, 655), f"Detected events: {summary['event_count']}   |   GPU pipeline: {summary['gpu_timings']['total_gpu_pipeline_s']:.4f}s", font=body_font, fill=MUTED)
    return [np.array(image)] * seconds_to_frames(4.0)


def annotated_video_section(project_root: Path, summary: dict) -> list[np.ndarray]:
    video_path = project_root / "data" / "output" / "annotated_motion_video.mp4"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, total_frames // 2 - 72)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    title_font = get_font(30, bold=True)
    body_font = get_font(22)
    frames_out: list[np.ndarray] = []

    for _ in range(min(144, total_frames - start_frame)):
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = canvas()
        draw = ImageDraw.Draw(pil)
        draw.text((60, 120), "Annotated output video", font=title_font, fill=TEXT)
        draw.text((60, 158), "The overlay shows frame index, motion score, and event labeling.", font=body_font, fill=MUTED)

        frame_img = Image.fromarray(frame_rgb)
        frame_img.thumbnail((1120, 500))
        x = (WIDTH - frame_img.width) // 2
        y = 200
        pil.paste(frame_img, (x, y))
        frames_out.append(np.array(pil))

    cap.release()
    return frames_out


def closing_slide(project_root: Path, summary: dict) -> list[np.ndarray]:
    image = canvas()
    draw = ImageDraw.Draw(image)
    title_font = get_font(38, bold=True)
    body_font = get_font(24)
    small_font = get_font(20)

    draw.text((60, 120), "Submission-ready deliverables", font=title_font, fill=TEXT)
    items = [
        "Public GitHub repository",
        "Execution log and benchmark summary",
        "CSV outputs and motion heatmap",
        "Annotated MP4 and sampled result image",
        "Presentation notes and demo script",
    ]
    y = 220
    for item in items:
        draw.rounded_rectangle((72, y + 8, 92, y + 28), radius=6, fill=ACCENT)
        draw.text((110, y), item, font=body_font, fill=TEXT)
        y += 58

    repo_text = "Project folder: gpu_capstone_motion_analytics"
    draw.text((60, 590), repo_text, font=small_font, fill=MUTED)
    draw.text((60, 622), f"Final GPU runtime: {summary['gpu_timings']['total_gpu_pipeline_s']:.4f}s", font=small_font, fill=MUTED)
    return [np.array(image)] * seconds_to_frames(3.5)


def write_video(frames: list[np.ndarray], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer for {output}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    args = parse_args()
    project_root = args.project_root
    summary_path = project_root / "artifacts" / "benchmark_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    log_lines = (project_root / "artifacts" / "execution_log.txt").read_text(encoding="utf-8").strip().splitlines()

    commands = [
        "python src/generate_synthetic_video.py --output-dir data/input --frames 720 --width 256 --height 256 --seed 22098",
        "python src/run_motion_analytics.py --input-dir data/input --output-dir data/output --artifacts-dir artifacts --presentation-dir presentation --batch-size 64 --cpu-subset 180",
    ]

    temp_chart = project_root / "presentation" / "_motion_chart.png"
    frames: list[np.ndarray] = []
    frames.extend(title_slide(summary))
    frames.extend(terminal_slide(commands, log_lines[:8]))
    frames.extend(collage_slide(project_root))
    frames.extend(chart_slide(project_root, summary, temp_chart))
    frames.extend(annotated_video_section(project_root, summary))
    frames.extend(closing_slide(project_root, summary))

    write_video(frames, args.output)

    if temp_chart.exists():
        temp_chart.unlink()

    manifest = {
        "video_path": str(args.output),
        "duration_seconds": round(len(frames) / FPS, 2),
        "fps": FPS,
        "resolution": [WIDTH, HEIGHT],
        "sections": [
            "title",
            "execution_flow",
            "input_output_collage",
            "motion_score_chart",
            "annotated_video_clip",
            "closing_summary",
        ],
    }
    (project_root / "presentation" / "DEMO_VIDEO_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
