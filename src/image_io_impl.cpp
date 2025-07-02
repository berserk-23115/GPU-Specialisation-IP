#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <cstring>
#include <algorithm>
#include "image_io.h"

namespace fs = std::filesystem;

// Load a single image from file
unsigned char* loadImage(const std::string& filename, int* width, int* height, int* channels) {
    // Read image using OpenCV
    cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
    
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << filename << std::endl;
        return nullptr;
    }
    
    // Set image dimensions
    *width = image.cols;
    *height = image.rows;
    *channels = image.channels();
    
    // Allocate memory for image data
    size_t imageSize = (*width) * (*height) * (*channels);
    unsigned char* imageData = new unsigned char[imageSize];
    
    // Copy image data
    std::memcpy(imageData, image.data, imageSize);
    
    return imageData;
}

// Save a single image to file
void saveImage(const std::string& filename, unsigned char* imageData, int width, int height, int channels) {
    // Create cv::Mat from image data
    cv::Mat image(height, width, CV_8UC(channels), imageData);
    
    // Save image using OpenCV
    bool success = cv::imwrite(filename, image);
    
    if (!success) {
        std::cerr << "Error: Failed to save image: " << filename << std::endl;
    }
}

// Load multiple images from a directory
std::vector<unsigned char*> loadImagesFromDirectory(
    const std::string& directory,
    std::vector<int>& widths,
    std::vector<int>& heights,
    std::vector<int>& channels
) {
    std::vector<unsigned char*> images;
    
    try {
        // Check if directory exists
        if (!fs::exists(directory) || !fs::is_directory(directory)) {
            std::cerr << "Error: Directory does not exist: " << directory << std::endl;
            return images;
        }
        
        // Iterate through directory
        for (const auto& entry : fs::directory_iterator(directory)) {
            // Check if entry is a file
            if (!fs::is_regular_file(entry)) {
                continue;
            }
            
            // Get file extension
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            // Check if file is an image
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" || 
                extension == ".bmp" || extension == ".tif" || extension == ".tiff") {
                
                // Load image
                int width, height, nChannels;
                unsigned char* imageData = loadImage(entry.path().string(), &width, &height, &nChannels);
                
                if (imageData != nullptr) {
                    // Add image to vectors
                    images.push_back(imageData);
                    widths.push_back(width);
                    heights.push_back(height);
                    channels.push_back(nChannels);
                    
                    std::cout << "Loaded: " << entry.path().filename() << " (" << width << "x" << height << ", " << nChannels << " channels)" << std::endl;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading images: " << e.what() << std::endl;
    }
    
    return images;
}

// Save multiple images to a directory
void saveImagesToDirectory(
    const std::string& directory,
    const std::vector<unsigned char*>& images,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& channels,
    const std::vector<std::string>& filenames
) {
    try {
        // Create directory if it does not exist
        if (!fs::exists(directory)) {
            fs::create_directories(directory);
        }
        
        // Save each image
        for (size_t i = 0; i < images.size(); ++i) {
            // Create full path
            std::string fullPath = directory + "/" + filenames[i];
            
            // Save image
            saveImage(fullPath, images[i], widths[i], heights[i], channels[i]);
            std::cout << "Saved: " << fullPath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error saving images: " << e.what() << std::endl;
    }
}

// Free memory allocated for an image
void freeImageMemory(unsigned char* imageData) {
    if (imageData != nullptr) {
        delete[] imageData;
    }
}

// Free memory allocated for multiple images
void freeImagesMemory(std::vector<unsigned char*>& images) {
    for (unsigned char* image : images) {
        freeImageMemory(image);
    }
    images.clear();
} 