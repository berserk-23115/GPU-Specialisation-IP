# CUDA Image Processor - Fixed Setup Guide

## Overview of Fixes Applied

The original codebase had C++ template compilation errors due to mixing C++17 features with CUDA compilation. Here's what was fixed:

### 1. **Separated C++ and CUDA Compilation**
- Created `src/image_io_impl.cpp` for pure C++ code with `std::filesystem`
- Modified `src/image_io.cu` to be a simple CUDA wrapper
- Updated `Makefile` to handle mixed compilation properly

### 2. **C++ Standard Compatibility**
- CUDA files use C++14 (better compatibility with nvcc)
- C++ files use C++17 (for `std::filesystem` support)
- Separated compilation prevents template conflicts

### 3. **Improved Build System**
- **Makefile**: Separate compilation rules for `.cu` and `.cpp` files
- **CMakeLists.txt**: Proper mixed C++/CUDA project configuration
- Better error handling and memory management

### 4. **Code Structure Improvements**
- Better argument parsing and validation
- Enhanced error handling and memory management
- More detailed logging and progress information
- Proper CUDA device detection and info display

## Prerequisites

1. **CUDA Toolkit** (tested with CUDA 11.0+)
   ```bash
   nvcc --version
   ```

2. **OpenCV 4** with development headers
   ```bash
   pkg-config --modversion opencv4
   ```

3. **G++ Compiler** with C++17 support
   ```bash
   g++ --version
   ```

## Build Instructions

### Option 1: Using Makefile (Recommended)
```bash
cd GPU-Specialisation-IP
./build.sh
```

### Option 2: Using CMake
```bash
cd GPU-Specialisation-IP
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Option 3: Manual Build
```bash
cd GPU-Specialisation-IP
mkdir -p obj bin
make clean
make -j$(nproc)
```

## Usage

```bash
./bin/cuda_image_processor --input data/input --output data/output --filter <filter_type>
```

### Available Filters:
- `blur` - Gaussian-like blur effect
- `sharpen` - Image sharpening
- `edge_detect` - Edge detection (Laplacian)
- `emboss` - Emboss effect

### Example:
```bash
# Create some test input
mkdir -p data/input data/output

# Run with blur filter
./bin/cuda_image_processor --input data/input --output data/output --filter blur

# Run with edge detection
./bin/cuda_image_processor --input data/input --output data/output --filter edge_detect
```

## File Structure

```
GPU-Specialisation-IP/
├── src/
│   ├── main.cu              # Main CUDA program
│   ├── kernels.cu           # CUDA kernel implementations
│   ├── image_io.cu          # CUDA wrapper for image I/O
│   └── image_io_impl.cpp    # Pure C++ image I/O implementation
├── include/
│   ├── kernels.h            # CUDA kernel headers
│   └── image_io.h           # Image I/O headers
├── data/
│   ├── input/               # Place input images here
│   └── output/              # Processed images saved here
├── Makefile                 # Build configuration
├── CMakeLists.txt          # Alternative build configuration
└── build.sh                # Build script
```

## Troubleshooting

### Common Issues:

1. **Template errors**: Fixed by separating C++17 filesystem code from CUDA compilation
2. **OpenCV warnings**: These are harmless OpenCV version warnings
3. **Architecture mismatch**: Adjust `-arch=sm_XX` in Makefile for your GPU

### GPU Compatibility:
- Default: `sm_60` (Pascal architecture)
- For newer GPUs: `sm_75` (Turing), `sm_80` (Ampere), `sm_86` (Ampere), etc.
- Check your GPU: `nvidia-smi` or `deviceQuery`

### Memory Requirements:
- Ensure sufficient GPU memory for your images
- Large images may require batch processing optimization

## Performance Notes

- Current implementation processes images sequentially
- For better performance, implement batch processing in `kernels.cu`
- Consider using CUDA streams for overlapping computation and memory transfers

## Testing

Place some test images (`.jpg`, `.png`, `.bmp`) in `data/input/` and run:

```bash
./bin/cuda_image_processor --input data/input --output data/output --filter blur
```

The processed images will appear in `data/output/` with descriptive filenames. 