FROM python:3.12-slim

WORKDIR /app

# 1. JAX CPU, memory management & numerical optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    JAX_FEM_LOG_LEVEL=ERROR \
    PYTHONPATH="/app" \
    FEM_TESSERACT_URL="http://fem_tesseract:8000" \
    GEOMETRY_TESSERACT_URL="http://geometry_tesseract:8001"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglu1-mesa \
    libgomp1 \
    libxrender1 \
    libxcursor1 \
    libxinerama1 \
    libxft2 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Optimized dependency caching layer
COPY REQUIREMENTS.txt /app/REQUIREMENTS.txt
RUN pip install --no-cache-dir -r /app/REQUIREMENTS.txt

# 3. Targeted directory copies (avoids copying git, images, tests)
COPY src/ /app/src/
COPY tesseracts/ /app/tesseracts/
COPY app.py /app/app.py

# 4. Pre-compile Python bytecode at build time
RUN python -m compileall /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
