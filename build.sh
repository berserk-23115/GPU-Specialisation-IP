#!/bin/bash

# Exit on error
set -e

# Create necessary directories
mkdir -p bin obj data/input data/output

# Check for CUDA
if command -v nvcc &> /dev/null; then
    echo "CUDA found, proceeding with build..."
else
    echo "CUDA not found! Please install CUDA toolkit and ensure nvcc is in your PATH."
    exit 1
fi

# Check for OpenCV
if pkg-config --exists opencv4; then
    echo "OpenCV found, proceeding with build..."
else
    echo "OpenCV not found! Please install OpenCV 4 and ensure pkg-config can find it."
    exit 1
fi

# Build the project using make
make clean
make -j$(nproc)

echo "Build completed successfully!"
echo "You can run the application with: ./bin/cuda_image_processor --input data/input --output data/output --filter blur" 