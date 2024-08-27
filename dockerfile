# Use the official Python image as the base image
FROM python:3.11-slim

# Set environment variables to non-interactively accept agreements for tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# If your project uses additional directories or files, copy them as well
COPY . /app

# Expose the port on which the Flask app will run
EXPOSE 5050

# Run the Flask application
CMD ["python", "app.py"]