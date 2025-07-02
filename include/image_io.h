#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#include <string>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

// Load a single image from file
unsigned char* loadImage(const std::string& filename, int* width, int* height, int* channels);

// Save a single image to file
void saveImage(const std::string& filename, unsigned char* imageData, int width, int height, int channels);

// Load multiple images from a directory
std::vector<unsigned char*> loadImagesFromDirectory(
    const std::string& directory,
    std::vector<int>& widths,
    std::vector<int>& heights,
    std::vector<int>& channels
);

// Save multiple images to a directory
void saveImagesToDirectory(
    const std::string& directory,
    const std::vector<unsigned char*>& images,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    const std::vector<int>& channels,
    const std::vector<std::string>& filenames
);

// Free memory allocated for images
void freeImageMemory(unsigned char* imageData);
void freeImagesMemory(std::vector<unsigned char*>& images);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_IO_H 