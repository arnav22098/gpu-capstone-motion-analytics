# Short Project Description

This capstone project explores GPU-accelerated motion analytics on a synthetic traffic-style video stream. I created a reproducible video dataset with 720 frames that includes normal traffic activity, a rush-hour traffic burst, and a pedestrian crossing event. The purpose of the project is to demonstrate how GPU-based processing can be used for a realistic temporal computer vision task instead of only a single-image operation.

The analytics pipeline uses CUDA-backed PyTorch tensor operations to process the frame sequence in batches. The GPU stages include grayscale conversion, Gaussian smoothing, frame differencing, threshold-based motion masking, and motion score aggregation. After the GPU computation finishes, the project writes frame-level motion scores to CSV, detects high-activity event windows, builds a motion heatmap, and creates an annotated output video.

I chose this project because it feels closer to a practical video analytics workload that could be relevant to smart transportation, robotics, or surveillance systems. The repository includes the full code, runnable command-line entry points, generated input and output data, proof-of-execution artifacts, and presentation support notes for a 5 to 10 minute demonstration.
