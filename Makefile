NVCC = nvcc
CXX = g++
OPENCV = `pkg-config --cflags --libs opencv4`
CUDA_PATH = /usr/local/cuda

NVCC_FLAGS = -std=c++17 -O3 -arch=sm_60 -Xcompiler -Wall,-Wextra,-fPIC
CXX_FLAGS = -std=c++17 -O3 -Wall -Wextra -fPIC

INCLUDES = -I$(CUDA_PATH)/include -Iinclude/ $(OPENCV)
LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcuda $(OPENCV)

SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRCS))
TARGET = $(BIN_DIR)/cuda_image_processor

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET) --input data/input --output data/output --filter blur

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Generate dependency files
deps: $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -M $^

# Note: This Makefile assumes CUDA and OpenCV are properly installed
# On many systems, you may need to adjust the paths to these libraries 