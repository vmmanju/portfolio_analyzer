FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2 and scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for file state
RUN mkdir -p data

# Expose Cloud Run default port
EXPOSE 8080

# Run Streamlit on port 8080 (Cloud Run requirement)
ENTRYPOINT ["streamlit", "run", "dashboard/run.py", \
    "--server.port=8080", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]

