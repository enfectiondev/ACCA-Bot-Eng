    # Use Python 3.12 base image
    FROM python:3.12

    # Install system dependencies
    RUN apt-get update && apt-get install -y \
        default-jdk \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

    # Set working directory
    WORKDIR /eng-chatbot

    # Copy requirements first to leverage Docker cache
    COPY requirements.txt .

    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt gunicorn uvicorn[standard]

    # Copy the rest of the application
    COPY . .

    # Set environment variables
    ENV PYTHONPATH=/eng-chatbot
    ENV PYTHONUNBUFFERED=1

    # Expose port 8001
    EXPOSE 8001

    # Command to run the application with Gunicorn using Uvicorn workers
    CMD ["gunicorn", "app:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001", "--timeout", "120"]