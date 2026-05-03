# Capstone Demo Outline

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
- Total GPU pipeline time: 0.2977 seconds.
- Detected events in the run: 2.
- The project shows how GPU acceleration can support large video-style analytics workloads.

## Slide 6 - Lessons learned
- Batched tensor operations make it practical to process long frame sequences efficiently.
- GPU-friendly pipelines are a strong fit for video analytics, surveillance, and robotics-style workloads.
