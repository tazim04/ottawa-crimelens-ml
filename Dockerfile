# Containerize the ML scoring script

FROM python:3.11-slim

# Set environment variables to prevent Python from writing .pyc files and to ensure that output is flushed immediately
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY score.py .
COPY train.py .

# Create a non-root user to run the application for better security
RUN useradd --create-home appuser
USER appuser

ENTRYPOINT ["python"]
CMD ["score.py"]
