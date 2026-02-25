FROM python:3.10-slim

WORKDIR /app

# Install CPU-only torch first (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create static folder and move index.html into it (FastAPI StaticFiles mount)
RUN mkdir -p static && \
    if [ -f index.html ]; then cp index.html static/index.html; fi

# Default command (overridden by docker-compose for ingest vs api)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]