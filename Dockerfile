# Multi-stage build for efficient image
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/root/.local/bin:$PATH

# Create app directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create data directory for outputs
RUN mkdir -p /app/data /app/outputs /app/configs

# Expose ports
EXPOSE 8501 8000

# Default command - can be overridden
CMD ["python", "main.py"]

# Alternative commands:
# For CLI: CMD ["python", "cli.py", "--help"]
# For Web: CMD ["streamlit", "run", "web_interface.py", "--server.port=8501", "--server.address=0.0.0.0"]
# For API: CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
