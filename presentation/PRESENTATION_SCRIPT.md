# Presentation Script

## Title

GPU-Accelerated Motion Analytics for Synthetic Traffic Video

## Demo flow

1. Introduce the problem:
   explain that the project analyzes a long frame sequence and detects periods of unusually high motion.
2. Show the input:
   mention the synthetic traffic video, the moving objects, and the two designed event windows.
3. Explain the GPU pipeline:
   grayscale conversion, Gaussian blur, temporal difference, motion mask creation, and event scoring.
4. Show the outputs:
   `motion_scores.csv`, `detected_events.csv`, `motion_heatmap.png`, `annotated_motion_video.mp4`, and `sample_motion_results.png`.
5. Summarize the result:
   emphasize that the project processed the entire frame sequence on the GPU and produced both quantitative and visual evidence.

## Questions to be ready for

- Why use the GPU instead of the CPU?
- Why batch the frames?
- How are motion events detected?
- How could this project be extended to real video data?
