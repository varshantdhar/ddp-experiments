# Use the official PyTorch image with CUDA and cuDNN (suitable for Voltage Park and most GPU VMs)
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Set environment variables for NCCL (recommended for multi-GPU)
ENV NCCL_DEBUG=INFO
ENV NCCL_P2P_DISABLE=1

# Expose any ports if needed (e.g., for TensorBoard)
# EXPOSE 6006

# Default command: run train_pipe.py with torchrun for 2 processes (adjust as needed)
CMD ["torchrun", "--nproc_per_node=2", "parallelism/train_pipe.py"] 