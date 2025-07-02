NVCC = nvcc
CXX = g++
OPENCV = `pkg-config --cflags --libs opencv4`
CUDA_PATH = /usr/local/cuda

# Separate flags for CUDA and C++ compilation
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_60 -Xcompiler -Wall,-Wextra,-fPIC
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC

INCLUDES = -I$(CUDA_PATH)/include -Iinclude/
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda $(OPENCV)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

# Separate CUDA and C++ source files
CUDA_SRCS = $(SRC_DIR)/kernels.cu $(SRC_DIR)/image_io.cu $(SRC_DIR)/main.cu
CPP_SRCS = $(SRC_DIR)/image_io_impl.cpp

CUDA_OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CUDA_SRCS))
CPP_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))

ALL_OBJS = $(CUDA_OBJS) $(CPP_OBJS)
TARGET = $(BIN_DIR)/cuda_image_processor

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(ALL_OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

# CUDA compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# C++ compilation rule
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(OPENCV) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET) --input data/input --output data/output --filter blur

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Generate dependency files
deps: $(CUDA_SRCS) $(CPP_SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -M $(CUDA_SRCS)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(OPENCV) -M $(CPP_SRCS)

# Debug build
debug: NVCC_FLAGS += -g -G
debug: CXX_FLAGS += -g
debug: $(TARGET)

# Note: This Makefile separates CUDA and C++ compilation to avoid conflicts

# Note: This Makefile assumes CUDA and OpenCV are properly installed
# On many systems, you may need to adjust the paths to these libraries 