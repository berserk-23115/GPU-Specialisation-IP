#!/bin/bash

# Test script for CUDA Image Processor
set -e

echo "=== CUDA Image Processor Test Script ==="

# Check if executable exists
if [ ! -f "bin/cuda_image_processor" ]; then
    echo "Error: Executable not found. Please run ./build.sh first."
    exit 1
fi

# Create test directories
mkdir -p data/input data/output

# Check if input images exist
if [ -z "$(ls -A data/input)" ]; then
    echo "Warning: No images found in data/input/"
    echo "Please add some test images (.jpg, .png, .bmp) to data/input/ directory"
    exit 1
fi

echo "Found images in data/input/:"
ls -la data/input/

# Test all filter types
filters=("blur" "sharpen" "edge_detect" "emboss")

for filter in "${filters[@]}"; do
    echo ""
    echo "=== Testing $filter filter ==="
    ./bin/cuda_image_processor --input data/input --output data/output --filter $filter
    
    if [ $? -eq 0 ]; then
        echo "✓ $filter filter completed successfully"
    else
        echo "✗ $filter filter failed"
    fi
done

echo ""
echo "=== Test completed ==="
echo "Check data/output/ for processed images"
ls -la data/output/ 