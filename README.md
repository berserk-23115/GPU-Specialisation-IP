# CUDA Image Processor

## Project Description
This project implements parallel image processing algorithms using CUDA for GPU acceleration. The application can process large batches of images or large-sized images efficiently by leveraging GPU parallelism.

## Features
- Image convolution operations (blur, edge detection, etc.)
- Batch processing for multiple images
- Support for different image sizes

## Requirements
- CUDA Toolkit (11.0 or later)
- OpenCV (for image I/O)
- C++ compiler compatible with CUDA
- CMake (3.10 or later)

## Building the Project
```bash
mkdir build && cd build
cmake ..
make
```

## Usage
```bash
./cuda_image_processor --input [input_directory] --output [output_directory] --filter [filter_type]
```

Filter types include:
- blur
- sharpen
- edge_detect
- emboss

## Project Structure
```
CUDAImageProcessor/
├── src/                 # Source files
│   ├── main.cu          # Main entry point
│   ├── image_io.cu      # Image loading/saving utilities
│   └── kernels.cu       # CUDA kernels for image processing
├── include/             # Header files
│   ├── image_io.h
│   └── kernels.h
├── data/                # Sample data directory
├── CMakeLists.txt       # Build configuration
└── README.md            # This file
```

## Implementation Details
The implementation focuses on using CUDA to parallelize image processing operations. Each pixel operation is handled by a separate thread on the GPU, allowing for efficient processing of large images or multiple images simultaneously.

Key optimizations include:
- Using shared memory for filter operations
- Coalesced memory access patterns
- Batch processing to maximize GPU utilization
- Optimized thread block sizes

## Performance Considerations
For optimal performance, the application adjusts thread block dimensions based on the GPU capabilities and image size. Memory transfers between CPU and GPU are minimized by processing batches of images when possible. 