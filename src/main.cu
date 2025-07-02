#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include "image_io.h"
#include "kernels.h"

// Function to print usage information
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " --input <input_dir> --output <output_dir> --filter <filter_type>" << std::endl;
    std::cout << "Filter types: blur, sharpen, edge_detect, emboss" << std::endl;
}

// Function to parse command line arguments
bool parseArguments(int argc, char** argv, std::string& inputDir, std::string& outputDir, std::string& filterType) {
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) return false;
        
        std::string arg = argv[i];
        
        if (arg == "--input") {
            inputDir = argv[i + 1];
        } else if (arg == "--output") {
            outputDir = argv[i + 1];
        } else if (arg == "--filter") {
            filterType = argv[i + 1];
        } else {
            return false;
        }
    }
    
    // Check if all required arguments are provided
    return !inputDir.empty() && !outputDir.empty() && !filterType.empty();
}

// Function to check CUDA device
bool checkCudaDevice() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices found! Error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    // Print CUDA device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    return true;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string inputDir, outputDir, filterType;
    
    if (!parseArguments(argc, argv, inputDir, outputDir, filterType)) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Print processing information
    std::cout << "=== CUDA Image Processor ===" << std::endl;
    std::cout << "Input directory: " << inputDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Filter type: " << filterType << std::endl;
    std::cout << "================================" << std::endl;
    
    // Check CUDA capabilities
    if (!checkCudaDevice()) {
        return 1;
    }
    
    // Load images from the input directory
    std::vector<int> widths, heights, channels;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "\nLoading images from: " << inputDir << std::endl;
    std::vector<unsigned char*> inputImages = loadImagesFromDirectory(inputDir, widths, heights, channels);
    
    if (inputImages.empty()) {
        std::cerr << "No images found in the input directory!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << inputImages.size() << " images" << std::endl;
    
    // Prepare output images and filenames
    std::vector<unsigned char*> outputImages(inputImages.size(), nullptr);
    std::vector<std::string> filenames;
    
    // Generate output filenames and allocate memory
    for (size_t i = 0; i < inputImages.size(); ++i) {
        filenames.push_back("processed_" + filterType + "_" + std::to_string(i) + ".png");
        
        // Allocate memory for output images
        size_t imageSize = widths[i] * heights[i] * channels[i];
        outputImages[i] = new unsigned char[imageSize];
        
        if (outputImages[i] == nullptr) {
            std::cerr << "Failed to allocate memory for output image " << i << std::endl;
            // Clean up previously allocated images
            for (size_t j = 0; j < i; ++j) {
                delete[] outputImages[j];
            }
            freeImagesMemory(inputImages);
            return 1;
        }
    }
    
    // Process images
    std::cout << "\nProcessing images with filter: " << filterType << "..." << std::endl;
    
    // Create a 3x3 filter based on the filter type
    float filter[9];
    generateFilter(filter, 3, filterType.c_str());
    
    // Process each image
    for (size_t i = 0; i < inputImages.size(); ++i) {
        std::cout << "Processing image " << (i + 1) << "/" << inputImages.size() << "..." << std::endl;
        
        // Apply convolution
        applyConvolution(
            inputImages[i],
            outputImages[i],
            filter,
            3,
            widths[i],
            heights[i],
            channels[i]
        );
    }
    
    // Save the processed images
    std::cout << "\nSaving processed images to: " << outputDir << std::endl;
    saveImagesToDirectory(outputDir, outputImages, widths, heights, channels, filenames);
    
    // Calculate and print processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "\n=== Processing completed in " << duration << " ms ===" << std::endl;
    
    // Free memory
    freeImagesMemory(inputImages);
    for (size_t i = 0; i < outputImages.size(); ++i) {
        delete[] outputImages[i];
    }
    
    return 0;
} 