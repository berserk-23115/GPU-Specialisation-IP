# Setup Guide

This document provides instructions for setting up and running the CUDA Image Processor project on different platforms.

## Prerequisites

- CUDA Toolkit 11.0 or later
- OpenCV 4.x
- C++17 compatible compiler (GCC 9+, MSVC 2019+)
- CMake 3.10 or later (for the CMake build option)
- Make (for the Makefile build option)

## Platform-specific Instructions

### Linux

1. Install CUDA Toolkit:
```bash
# Check available versions in your package manager
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

2. Install OpenCV:
```bash
sudo apt install libopencv-dev
```

3. Build using the provided script:
```bash
./build.sh
```

### Windows

1. Install CUDA Toolkit:
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation wizard instructions

2. Install OpenCV:
   - Download from [OpenCV Releases](https://opencv.org/releases/)
   - Extract to a folder (e.g., C:\opencv)
   - Add the bin directory to your PATH (e.g., C:\opencv\build\x64\vc15\bin)

3. Build using CMake:
```
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### macOS

Note: CUDA is not officially supported on macOS since macOS 10.14 (Mojave). These instructions are for reference only or for users with older macOS versions.

1. Install dependencies via Homebrew:
```bash
brew install cmake opencv
```

2. Install CUDA (legacy only):
   - Download a compatible version from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow the installation instructions

3. Build using CMake:
```bash
mkdir build
cd build
cmake ..
make
```

## Using Docker

For cross-platform development, you can use Docker with NVIDIA Container Toolkit:

1. Install Docker and NVIDIA Container Toolkit:
```bash
# Install Docker (Ubuntu example)
sudo apt install docker.io

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

2. Build and run the Docker container:
```bash
# Build the Docker image
docker build -t cuda_image_processor .

# Run the container with GPU support
docker run --gpus all -it --rm cuda_image_processor
```

## Running the Application

After building, run the application with:
```bash
./bin/cuda_image_processor --input data/input --output data/output --filter blur
```

Or use the provided script to run all filters:
```bash
./run_all_filters.sh
``` 