FROM python:3.9-slim

# Set working dir in container
WORKDIR /Docker

# Copy your code into the image
COPY . /Docker

# If/when you add dependencies, uncomment:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command—replace app.py with your main script
CMD ["python", "Docker.py"]
