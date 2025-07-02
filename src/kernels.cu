#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <string.h>
#include "kernels.h"

// Define block size for CUDA kernels
#define BLOCK_SIZE 16

// CUDA error checking macro
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// CUDA kernel for image convolution
__global__ void convolutionKernel(
    const unsigned char* inputImage,
    unsigned char* outputImage,
    const float* filter,
    int filterWidth,
    int imageWidth,
    int imageHeight,
    int imageChannels
) {
    // Calculate pixel position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if the thread is within image boundaries
    if (x < imageWidth && y < imageHeight) {
        // Calculate half filter size
        int halfFilterWidth = filterWidth / 2;
        
        // Process each channel
        for (int c = 0; c < imageChannels; c++) {
            float sum = 0.0f;
            
            // Apply filter
            for (int fy = 0; fy < filterWidth; fy++) {
                for (int fx = 0; fx < filterWidth; fx++) {
                    // Calculate source pixel position
                    int sourceX = x + (fx - halfFilterWidth);
                    int sourceY = y + (fy - halfFilterWidth);
                    
                    // Clamp source position to image boundaries
                    sourceX = max(0, min(sourceX, imageWidth - 1));
                    sourceY = max(0, min(sourceY, imageHeight - 1));
                    
                    // Calculate source index
                    int sourceIndex = (sourceY * imageWidth + sourceX) * imageChannels + c;
                    
                    // Get pixel value and apply filter
                    float pixelValue = static_cast<float>(inputImage[sourceIndex]);
                    sum += pixelValue * filter[fy * filterWidth + fx];
                }
            }
            
            // Clamp result to [0, 255]
            sum = max(0.0f, min(sum, 255.0f));
            
            // Write result to output image
            int outputIndex = (y * imageWidth + x) * imageChannels + c;
            outputImage[outputIndex] = static_cast<unsigned char>(sum);
        }
    }
}

// Generate different types of filters
void generateFilter(float* filter, int filterWidth, const char* filterType) {
    // Reset filter
    memset(filter, 0, filterWidth * filterWidth * sizeof(float));
    
    if (strcmp(filterType, "blur") == 0) {
        // Create a simple box blur filter
        float value = 1.0f / (filterWidth * filterWidth);
        for (int i = 0; i < filterWidth * filterWidth; i++) {
            filter[i] = value;
        }
    } else if (strcmp(filterType, "sharpen") == 0) {
        // Create a sharpen filter
        // Center value is positive, surrounding values are negative
        int center = filterWidth / 2;
        for (int y = 0; y < filterWidth; y++) {
            for (int x = 0; x < filterWidth; x++) {
                filter[y * filterWidth + x] = -0.15f;
            }
        }
        filter[center * filterWidth + center] = 2.0f; // Center pixel
    } else if (strcmp(filterType, "edge_detect") == 0) {
        // Create a simple edge detection filter (Laplacian)
        int center = filterWidth / 2;
        for (int y = 0; y < filterWidth; y++) {
            for (int x = 0; x < filterWidth; x++) {
                filter[y * filterWidth + x] = -1.0f;
            }
        }
        filter[center * filterWidth + center] = filterWidth * filterWidth - 1.0f; // Center pixel
    } else if (strcmp(filterType, "emboss") == 0) {
        // Create an emboss filter
        filter[0] = -2.0f; filter[1] = -1.0f; filter[2] = 0.0f;
        filter[3] = -1.0f; filter[4] = 1.0f;  filter[5] = 1.0f;
        filter[6] = 0.0f;  filter[7] = 1.0f;  filter[8] = 2.0f;
    } else {
        // Default to identity filter
        int center = filterWidth / 2;
        filter[center * filterWidth + center] = 1.0f;
    }
}

// Host wrapper function for convolution kernel
void applyConvolution(
    const unsigned char* h_inputImage,
    unsigned char* h_outputImage,
    const float* h_filter,
    int filterWidth,
    int imageWidth,
    int imageHeight,
    int imageChannels
) {
    // Allocate device memory
    unsigned char *d_inputImage, *d_outputImage;
    float *d_filter;
    size_t imageSize = imageWidth * imageHeight * imageChannels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);
    
    // Allocate device memory
    cudaCheckError(cudaMalloc(&d_inputImage, imageSize));
    cudaCheckError(cudaMalloc(&d_outputImage, imageSize));
    cudaCheckError(cudaMalloc(&d_filter, filterSize));
    
    // Copy data from host to device
    cudaCheckError(cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_filter, h_filter, filterSize, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((imageWidth + blockDim.x - 1) / blockDim.x, 
                 (imageHeight + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    convolutionKernel<<<gridDim, blockDim>>>(
        d_inputImage,
        d_outputImage,
        d_filter,
        filterWidth,
        imageWidth,
        imageHeight,
        imageChannels
    );
    
    // Check for kernel launch errors
    cudaCheckError(cudaGetLastError());
    
    // Wait for kernel to complete
    cudaCheckError(cudaDeviceSynchronize());
    
    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaCheckError(cudaFree(d_inputImage));
    cudaCheckError(cudaFree(d_outputImage));
    cudaCheckError(cudaFree(d_filter));
}

// Batch process multiple images
void batchProcessImages(
    const unsigned char** h_inputImages,
    unsigned char** h_outputImages,
    int numImages,
    int imageWidth,
    int imageHeight,
    int imageChannels,
    const char* filterType
) {
    // Create filter
    int filterWidth = 3; // Using a 3x3 filter
    float h_filter[9];
    generateFilter(h_filter, filterWidth, filterType);
    
    // Process each image
    for (int i = 0; i < numImages; i++) {
        applyConvolution(
            h_inputImages[i],
            h_outputImages[i],
            h_filter,
            filterWidth,
            imageWidth,
            imageHeight,
            imageChannels
        );
    }
    
    // Note: In a real implementation, you would process images in batches to maximize GPU utilization
    // This would involve more complex memory management and kernel design
} 