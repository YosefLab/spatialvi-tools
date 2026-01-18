# Dockerfile for spatialvi-tools
# Multi-stage build for optimized image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install package and dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel && \
    pip install --no-cache-dir dist/*.whl

# Runtime stage
FROM python:3.11-slim as runtime

LABEL maintainer="YosefLab"
LABEL description="spatialvi-tools: Unified toolbox for spatial transcriptomics analysis"
LABEL version="0.1.0"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN useradd --create-home --shell /bin/bash spatialvi
USER spatialvi

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command
CMD ["python", "-c", "import spatialvi; print(f'spatialvi-tools version: {spatialvi.__version__}')"]

# CUDA variant
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as cuda

LABEL maintainer="YosefLab"
LABEL description="spatialvi-tools with CUDA support"

# Install Python and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install with CUDA-enabled PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash spatialvi
USER spatialvi

ENV PYTHONUNBUFFERED=1

CMD ["python", "-c", "import spatialvi; import torch; print(f'spatialvi-tools: {spatialvi.__version__}, CUDA: {torch.cuda.is_available()}')"]
