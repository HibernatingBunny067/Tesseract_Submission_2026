FROM python:3.12-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY tesseracts/geometry_tesseract/tesseract_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY tesseracts /app/tesseracts

ENV PYTHONPATH="/app"

CMD ["python", "tesseracts/geometry_tesseract/tesseract_server.py"]
