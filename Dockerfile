FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

# Set noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopencv-dev \
    pkg-config \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p bin obj data/input data/output

# Build the application
RUN make

# Set the entrypoint
ENTRYPOINT ["./bin/cuda_image_processor"]

# Default command (can be overridden)
CMD ["--input", "data/input", "--output", "data/output", "--filter", "blur"] 