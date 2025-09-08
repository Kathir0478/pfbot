# Base Python image
FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Render will override with $PORT)
EXPOSE 5000

# Run app with Gunicorn
CMD ["gunicorn", "-w", "1", "-k", "sync", "-t", "120", "-b", "0.0.0.0:5000", "app:app"]
