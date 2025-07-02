#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

// Kernel function to apply convolution to an image
__global__ void convolutionKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    const float* filter,
    int filterWidth,
    int imageWidth,
    int imageHeight,
    int imageChannels
);

// Host wrapper function for convolution kernel
void applyConvolution(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    const float* h_filter,
    int filterWidth,
    int imageWidth,
    int imageHeight,
    int imageChannels
);

// Generate different types of filters
void generateFilter(float* filter, int filterWidth, const char* filterType);

// Batch process multiple images
void batchProcessImages(
    const unsigned char** h_inputImages,
    unsigned char** h_outputImages,
    int numImages,
    int imageWidth,
    int imageHeight,
    int imageChannels,
    const char* filterType
);

#endif // KERNELS_H 