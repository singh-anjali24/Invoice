# ── Stage 1: Build ──────────────────────────────────────────────────────
FROM python:3.13-slim AS base

# Install system dependencies (Tesseract + Poppler) — no manual downloads needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Generate sample invoices at build time
RUN python scripts/generate_samples.py

# Expose the API port
EXPOSE 8000

# Default command: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
