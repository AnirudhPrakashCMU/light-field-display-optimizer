# GPU base for CUDA-accelerated light field optimization
FROM runpod/serverless:cuda11.8.0-py3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Tell RunPod where the handler lives
ENV HANDLER_PATH=rp_handler
ENV RP_HANDLER=handler

# Start the serverless worker
CMD ["bash", "/app/start.sh"]