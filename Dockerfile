FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py /app/

# Create input and output directories
RUN mkdir -p /app/input
RUN mkdir -p /app/output

# Set the command to run the application
CMD ["python", "pdf_extractor.py", "/app/input", "/app/output"]
