# Scoring API. Build the bundle first:  uv run python -m fraud.serving.bundle
#   docker build -t fraud-scorer .
#   docker run -p 8000:8000 fraud-scorer
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY src/ src/
COPY artifacts/bundle/ artifacts/bundle/

ENV PYTHONPATH=/app/src \
    FRAUD_BUNDLE=/app/artifacts/bundle \
    PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fraud.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
