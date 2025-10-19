# ANPR System Docker Container
# Multi-stage build for optimized production image

# Stage 1: Base image with Python and system dependencies
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build stage with development dependencies
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional web dependencies
RUN pip install --no-cache-dir streamlit streamlit-option-menu plotly

# Stage 3: Production image
FROM base as production

# Create non-root user
RUN groupadd -r anpr && useradd -r -g anpr anpr

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY anpr_system/ ./anpr_system/
COPY runs/ ./runs/
COPY utility_files/ ./utility_files/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .

# Clone YOLOv5
RUN git clone https://github.com/ultralytics/yolov5.git

# Create necessary directories
RUN mkdir -p results test_data

# Set permissions
RUN chown -R anpr:anpr /app

# Switch to non-root user
USER anpr

# Expose port for web interface
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import anpr_system; print('OK')" || exit 1

# Default command (web interface)
CMD ["streamlit", "run", "anpr_system/web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Alternative commands:
# For CLI: docker run anpr-system anpr --help
# For web: docker run -p 8501:8501 anpr-system
