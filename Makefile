PYTHON ?= python3
INPUT_DIR := data/input
OUTPUT_DIR := data/output
ARTIFACTS_DIR := artifacts
PRESENTATION_DIR := presentation
FRAMES ?= 720
WIDTH ?= 256
HEIGHT ?= 256
BATCH_SIZE ?= 64
CPU_SUBSET ?= 180

.PHONY: all build dataset run clean help

all: run

build:
	@echo "Install dependencies with: pip install -r requirements.txt"

dataset:
	$(PYTHON) src/generate_synthetic_video.py --output-dir $(INPUT_DIR) --frames $(FRAMES) --width $(WIDTH) --height $(HEIGHT) --seed 22098

run: dataset
	$(PYTHON) src/run_motion_analytics.py --input-dir $(INPUT_DIR) --output-dir $(OUTPUT_DIR) --artifacts-dir $(ARTIFACTS_DIR) --presentation-dir $(PRESENTATION_DIR) --batch-size $(BATCH_SIZE) --cpu-subset $(CPU_SUBSET)

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(str(p)) if p.is_dir() else p.unlink() for base in ['$(OUTPUT_DIR)', '$(ARTIFACTS_DIR)', '$(INPUT_DIR)', '$(PRESENTATION_DIR)'] for p in Path(base).glob('*')]"

help:
	@echo "make dataset   - generate the synthetic video dataset"
	@echo "make run       - generate data and run the GPU analytics pipeline"
	@echo "make clean     - remove generated inputs, outputs, and artifacts"
