FROM python:3.12-slim

WORKDIR /app

# 1. JAX CPU, FEM & PETSc threading optimization environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    XLA_FLAGS="--xla_cpu_multi_thread_eigen=false" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 \
    JAX_PLATFORMS=cpu \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    PETSC_OPTIONS="-ksp_type preonly -pc_type lu" \
    JAX_FEM_LOG_LEVEL=ERROR \
    TESSERACT_OUTPUT_PATH="/tmp/tesseract_runs" \
    PYTHONPATH="/app"

# Install system dependencies for FEM, Gmsh, PETSc, and OpenGL
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
COPY tesseracts/fem_tesseract/tesseract_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 3. Targeted directory copies
COPY src/ /app/src/
COPY tesseracts/ /app/tesseracts/

# 4. Pre-compile Python bytecode at build time
RUN python -m compileall /app

EXPOSE 8000

CMD ["python", "tesseracts/fem_tesseract/tesseract_server.py"]
