# Official PyTorch image with CUDA + cuDNN support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Upgrade pip
RUN pip install --upgrade pip

# Install additional libraries
RUN pip install \
    torchvision \
    transformers

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Default command
CMD ["python", "your_script.py"]