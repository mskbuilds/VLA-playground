FROM nvcr.io/nvidia/l4t-base:r36.2.0
# Install any extra Python libraries you need
# Install Python & dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install PyTorch wheel for Jetson (with CUDA)
RUN pip3 install --no-cache-dir \
    https://nvidia-ai-iot.github.io/pytorch-for-jetson/l4t-36/pytorch-2.1.0-cp310-cp310-linux_aarch64.whl && \
    pip3 install --no-cache-dir torchvision transformers

    
WORKDIR /app
COPY . /app

# CMD ["python3", "your_script.py"]