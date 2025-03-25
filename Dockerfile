FROM python:3.10-slim

# Install necessary dependencies
RUN apt-get update && apt-get install -y git && apt-get clean
RUN pip install --no-cache-dir \
    torch \
    transformers \
    datasets \
    accelerate \
    peft \
    bitsandbytes \
    sentencepiece \
    scikit-learn \
    google-cloud-storage \
    huggingface-hub

# Set working directory inside the container
WORKDIR /app

# Copy model from your local machine or VM into Docker image
COPY /path/to/model /app/model/

# Copy script
COPY train_finn_phased_lora.py .

# Set entrypoint for the container
ENTRYPOINT ["python", "train_finn_phased_lora.py"]