from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def draw_vehicle(frame: np.ndarray, center_x: float, center_y: float, width: int, height: int, color: tuple[int, int, int]) -> None:
    x0 = int(max(0, center_x - width // 2))
    y0 = int(max(0, center_y - height // 2))
    x1 = int(min(frame.shape[1] - 1, center_x + width // 2))
    y1 = int(min(frame.shape[0] - 1, center_y + height // 2))
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=-1)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (245, 245, 245), thickness=1)


def generate_frames(output_dir: Path, frames: int, width: int, height: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    video_path = output_dir / "synthetic_traffic.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        24.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not open video writer for synthetic_traffic.mp4")

    metadata_path = output_dir / "input_metadata.csv"
    frame_array_path = output_dir / "frames.npy"

    all_frames = np.empty((frames, height, width, 3), dtype=np.uint8)

    vehicles = [
        {"lane_y": int(height * 0.28), "x": -20.0, "speed": 2.8, "size": (26, 14), "color": (50, 180, 255)},
        {"lane_y": int(height * 0.44), "x": width + 30.0, "speed": -3.4, "size": (32, 16), "color": (80, 255, 120)},
        {"lane_y": int(height * 0.61), "x": -50.0, "speed": 4.2, "size": (28, 14), "color": (255, 120, 80)},
        {"lane_y": int(height * 0.77), "x": width + 60.0, "speed": -2.4, "size": (34, 18), "color": (255, 210, 70)},
    ]

    with metadata_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer_csv = csv.writer(csv_file)
        writer_csv.writerow(["frame_index", "active_objects", "event_label", "avg_brightness"])

        for frame_index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)

            gradient_x = np.linspace(25, 80, width, dtype=np.uint8)
            frame[:, :, 0] = gradient_x
            frame[:, :, 1] = gradient_x[::-1]
            frame[:, :, 2] = 55

            for lane in [0.25, 0.40, 0.57, 0.73]:
                y = int(height * lane)
                cv2.line(frame, (0, y), (width, y), (115, 115, 115), 2)
                for dash_x in range(0, width, 28):
                    cv2.line(frame, (dash_x, y + 12), (dash_x + 14, y + 12), (210, 210, 210), 1)

            noise = rng.integers(0, 10, size=frame.shape, dtype=np.uint8)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)

            event_label = "normal"
            active_objects = 0

            for vehicle in vehicles:
                vehicle["x"] += vehicle["speed"]
                if vehicle["speed"] > 0 and vehicle["x"] - vehicle["size"][0] > width:
                    vehicle["x"] = -rng.integers(40, 120)
                elif vehicle["speed"] < 0 and vehicle["x"] + vehicle["size"][0] < 0:
                    vehicle["x"] = width + rng.integers(40, 120)

                if 0 - vehicle["size"][0] <= vehicle["x"] <= width + vehicle["size"][0]:
                    draw_vehicle(frame, vehicle["x"], vehicle["lane_y"], vehicle["size"][0], vehicle["size"][1], vehicle["color"])
                    active_objects += 1

            if 220 <= frame_index <= 280:
                event_label = "rush_hour_wave"
                for offset in range(3):
                    burst_x = (frame_index * 6 + offset * 40) % (width + 80) - 40
                    burst_y = int(height * (0.22 + 0.18 * offset))
                    draw_vehicle(frame, burst_x, burst_y, 22 + 2 * offset, 12 + offset, (255, 80 + offset * 30, 180))
                    active_objects += 1

            if 500 <= frame_index <= 560:
                event_label = "pedestrian_crossing"
                for person in range(6):
                    cx = 30 + person * 34 + (frame_index - 500) * 1.5
                    cy = int(height * 0.5 + (-1) ** person * 16)
                    cv2.circle(frame, (int(cx), int(cy)), 7, (240, 240, 240), thickness=-1)
                    cv2.line(frame, (int(cx), int(cy + 7)), (int(cx), int(cy + 20)), (240, 240, 240), 2)
                active_objects += 6

            avg_brightness = float(frame.mean())
            writer.write(frame)
            all_frames[frame_index] = frame

            writer_csv.writerow([frame_index, active_objects, event_label, f"{avg_brightness:.3f}"])

            if frame_index in {0, frames // 3, 2 * frames // 3, frames - 1}:
                cv2.imwrite(str(output_dir / f"input_frame_{frame_index:04d}.png"), frame)

    writer.release()
    np.save(frame_array_path, all_frames)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic traffic video dataset for GPU motion analytics.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store the synthetic video and frame data.")
    parser.add_argument("--frames", type=int, default=720, help="Number of frames in the synthetic video.")
    parser.add_argument("--width", type=int, default=256, help="Frame width in pixels.")
    parser.add_argument("--height", type=int, default=256, help="Frame height in pixels.")
    parser.add_argument("--seed", type=int, default=22098, help="Random seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_frames(args.output_dir, args.frames, args.width, args.height, args.seed)
    print(f"Generated synthetic video dataset with {args.frames} frames in {args.output_dir}")


if __name__ == "__main__":
    main()
