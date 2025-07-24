# Use Python 3.10 slim image for optimal performance
FROM python:3.10.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV DISPLAY=:99
ENV MPLBACKEND=Agg

# Install system dependencies required for chemistry packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++ \
    curl \
    libboost-all-dev \
    libeigen3-dev \
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libxft2 \
    libx11-6 \
    libcairo2-dev \
    libgirepository1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip wheel setuptools

# Copy requirements first (for better Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create .streamlit directory and copy config
RUN mkdir -p .streamlit
COPY .streamlit/config.toml .streamlit/

# Set proper permissions and create startup script
RUN chmod +x /app && \
    echo '#!/bin/bash\nexport MPLBACKEND=Agg\nexport DISPLAY=:99\nstreamlit run main_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["/app/start.sh"]
