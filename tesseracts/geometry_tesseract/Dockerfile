FROM python:3.12-slim

WORKDIR /app

# 1. JAX CPU & numerical optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PYTHONPATH="/app"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY tesseracts/geometry_tesseract/tesseract_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/fem/ /app/src/fem/
COPY tesseracts/geometry_tesseract/ /app/tesseracts/geometry_tesseract/

RUN python -m compileall /app

EXPOSE 8001

CMD ["python", "tesseracts/geometry_tesseract/tesseract_server.py"]
