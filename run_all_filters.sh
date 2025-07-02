#!/bin/bash

# Path to executable
EXECUTABLE="./bin/cuda_image_processor"

# Input and output directories
INPUT_DIR="data/input"
OUTPUT_DIR="data/output"

# Available filters
FILTERS=("blur" "sharpen" "edge_detect" "emboss")

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found at $EXECUTABLE"
    echo "Please build the project first using ./build.sh"
    exit 1
fi

# Check if input directory has images
IMAGE_COUNT=$(ls -1 "$INPUT_DIR"/*.{jpg,jpeg,png,bmp} 2>/dev/null | wc -l)
if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo "No images found in $INPUT_DIR"
    echo "Please add some images to the input directory first."
    exit 1
fi

# Create output subdirectories for each filter
for FILTER in "${FILTERS[@]}"; do
    mkdir -p "$OUTPUT_DIR/$FILTER"
done

# Process images with each filter
for FILTER in "${FILTERS[@]}"; do
    echo "Processing images with filter: $FILTER"
    
    # Run the image processor with the current filter
    "$EXECUTABLE" --input "$INPUT_DIR" --output "$OUTPUT_DIR/$FILTER" --filter "$FILTER"
    
    echo "Completed processing with $FILTER filter"
    echo "Results saved to $OUTPUT_DIR/$FILTER"
    echo
done

echo "All filters completed!"
echo "Check the output directory for results." 