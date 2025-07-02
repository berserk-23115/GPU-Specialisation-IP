cmake_minimum_required(VERSION 3.10)
project(CUDAImageProcessor LANGUAGES CXX CUDA)

# CUDA Configuration
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# OpenCV Configuration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Set CUDA architecture flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")  # Adjust based on your GPU

# Executable definition
cuda_add_executable(
    cuda_image_processor
    src/main.cu
    src/image_io.cu
    src/kernels.cu
)

# Link libraries
target_link_libraries(cuda_image_processor ${CUDA_LIBRARIES} ${OpenCV_LIBS})

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR}) 