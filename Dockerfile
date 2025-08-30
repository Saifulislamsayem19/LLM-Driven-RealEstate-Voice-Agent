# Use a small, modern Python image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (OpenBLAS/OMP helpful for faiss-cpu; ffmpeg for audio ops)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libopenblas-dev libomp-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt gunicorn

# Copy the app
COPY . /app

# Spaces serve on 7860; expose and set env just in case
ENV PORT=7860
EXPOSE 7860

# Make FAISS/OpenMP behave well on small containers
ENV OMP_NUM_THREADS=1

# Start the Flask app via Gunicorn: module:variable -> app:app
CMD ["gunicorn", "-k", "gthread", "--threads", "2", "-b", "0.0.0.0:7860", "app:app"]
