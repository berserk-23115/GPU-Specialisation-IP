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
        std::string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            inputDir = argv[i + 1];
        } else if (arg == "--output" && i + 1 < argc) {
            outputDir = argv[i + 1];
        } else if (arg == "--filter" && i + 1 < argc) {
            filterType = argv[i + 1];
        } else {
            return false;
        }
    }
    
    // Check if all required arguments are provided
    return !inputDir.empty() && !outputDir.empty() && !filterType.empty();
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string inputDir, outputDir, filterType;
    
    if (!parseArguments(argc, argv, inputDir, outputDir, filterType)) {
        printUsage(argv[0]);
        return 1;
    }
    
    // Print processing information
    std::cout << "CUDA Image Processor" << std::endl;
    std::cout << "Input directory: " << inputDir << std::endl;
    std::cout << "Output directory: " << outputDir << std::endl;
    std::cout << "Filter type: " << filterType << std::endl;
    
    // Check CUDA capabilities
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found! Exiting..." << std::endl;
        return 1;
    }
    
    // Print CUDA device information
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    std::cout << "Using CUDA device: " << deviceProp.name << std::endl;
    
    // Load images from the input directory
    std::vector<int> widths, heights, channels;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::cout << "Loading images..." << std::endl;
    std::vector<unsigned char*> inputImages = loadImagesFromDirectory(inputDir, widths, heights, channels);
    
    if (inputImages.empty()) {
        std::cerr << "No images found in the input directory!" << std::endl;
        return 1;
    }
    
    std::cout << "Loaded " << inputImages.size() << " images" << std::endl;
    
    // Prepare output images
    std::vector<unsigned char*> outputImages(inputImages.size(), nullptr);
    std::vector<std::string> filenames;
    
    // Extract filenames from input directory
    // In a real implementation, this would extract proper filenames
    for (size_t i = 0; i < inputImages.size(); ++i) {
        filenames.push_back("processed_" + std::to_string(i) + ".png");
        
        // Allocate memory for output images
        outputImages[i] = new unsigned char[widths[i] * heights[i] * channels[i]];
    }
    
    // Process images in batch
    std::cout << "Processing images with filter: " << filterType << "..." << std::endl;
    
    // Convert std::vector to raw pointers for CUDA processing
    unsigned char** d_inputImages = new unsigned char*[inputImages.size()];
    unsigned char** d_outputImages = new unsigned char*[outputImages.size()];
    
    for (size_t i = 0; i < inputImages.size(); ++i) {
        d_inputImages[i] = inputImages[i];
        d_outputImages[i] = outputImages[i];
    }
    
    // Batch process images (this would call CUDA kernels)
    for (size_t i = 0; i < inputImages.size(); ++i) {
        // Example of processing each image
        // In a real implementation, this would be batched for efficiency
        
        // Create a 3x3 filter based on the filter type
        float filter[9];
        generateFilter(filter, 3, filterType.c_str());
        
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
    
    // Clean up temporary arrays
    delete[] d_inputImages;
    delete[] d_outputImages;
    
    // Save the processed images
    std::cout << "Saving processed images..." << std::endl;
    saveImagesToDirectory(outputDir, outputImages, widths, heights, channels, filenames);
    
    // Calculate and print processing time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    std::cout << "Processing completed in " << duration << " ms" << std::endl;
    
    // Free memory
    freeImagesMemory(inputImages);
    freeImagesMemory(outputImages);
    
    return 0;
} 