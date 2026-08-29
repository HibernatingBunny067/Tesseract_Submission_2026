FROM python:3.12-slim

WORKDIR /app

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

# Install FEM-specific Python dependencies
COPY tesseracts/fem_tesseract/tesseract_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source tree
COPY src /app/src
COPY tesseracts /app/tesseracts

ENV PYTHONPATH="/app"
ENV JAX_FEM_LOG_LEVEL="ERROR"
ENV TESSERACT_OUTPUT_PATH="/tmp/tesseract_runs"

CMD ["python", "tesseracts/fem_tesseract/tesseract_server.py"]
