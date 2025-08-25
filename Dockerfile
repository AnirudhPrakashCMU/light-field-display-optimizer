# GPU base for CUDA-accelerated light field optimization
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install RunPod serverless SDK first
RUN pip install --no-cache-dir runpod

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set environment variables for GPU optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONUNBUFFERED=1

# Tell RunPod where the handler lives
ENV HANDLER_PATH=rp_handler
ENV RP_HANDLER=handler

# Start the serverless worker
CMD ["python", "-u", "rp_handler.py"]