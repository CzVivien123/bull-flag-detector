FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only PyTorch
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/run_pipeline.py"]
