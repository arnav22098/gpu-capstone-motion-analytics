$ErrorActionPreference = "Stop"

python src/generate_synthetic_video.py `
  --output-dir data/input `
  --frames 720 `
  --width 256 `
  --height 256 `
  --seed 22098

python src/run_motion_analytics.py `
  --input-dir data/input `
  --output-dir data/output `
  --artifacts-dir artifacts `
  --presentation-dir presentation `
  --batch-size 64 `
  --cpu-subset 180
