FROM python:3.12-slim

WORKDIR /app

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

# Install all project dependencies
COPY REQUIREMENTS.txt /app/REQUIREMENTS.txt
RUN pip install --no-cache-dir -r /app/REQUIREMENTS.txt

# Copy codebase
COPY . /app

ENV PYTHONPATH="/app"
ENV JAX_FEM_LOG_LEVEL="ERROR"
ENV FEM_TESSERACT_URL="http://fem_tesseract:8000"
ENV GEOMETRY_TESSERACT_URL="http://geometry_tesseract:8001"

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true"]
