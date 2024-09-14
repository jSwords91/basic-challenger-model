FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code (from --> to)
COPY . .

# Create logs directory
RUN mkdir -p logs models

VOLUME ["/app/logs", "/app/models"]


CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]