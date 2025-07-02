cmake_minimum_required(VERSION 3.10)
project(CUDAImageProcessor LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standard (use C++14 for better compatibility)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA Configuration
find_package(CUDA REQUIRED)
enable_language(CUDA)

# OpenCV Configuration
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# Set CUDA architecture flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Create executable with mixed C++/CUDA sources
add_executable(cuda_image_processor
    src/main.cu
    src/image_io.cu
    src/kernels.cu
    src/image_io_impl.cpp
)

# Set target properties
set_target_properties(cuda_image_processor PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CUDA_STANDARD 14
)

# Link libraries
target_link_libraries(cuda_image_processor 
    ${CUDA_LIBRARIES} 
    ${OpenCV_LIBS}
)

# Copy data directory to build directory
file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR}) 