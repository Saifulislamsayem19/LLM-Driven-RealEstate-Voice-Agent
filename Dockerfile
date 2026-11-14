FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# All pinned dependencies ship prebuilt wheels for Python 3.14, so no compiler toolchain is needed
COPY requirements.txt .
RUN pip install --only-binary=:all: -r requirements.txt

COPY . .

# Writable at runtime even when the container runs as a non-root user (e.g. Hugging Face Spaces)
RUN mkdir -p vector_db user_data && chmod -R a+rwX vector_db user_data

EXPOSE 7860

# /status returns 503 until the vector DB and LLM chain are ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/status', timeout=5)" || exit 1

# Behind a reverse proxy (e.g. Hugging Face Spaces): use the real client IP for rate limiting
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
