# OnTime scoring API. Build from the repo root AFTER `make all` (models and features are not in git):
#   docker build -t ontime-api .   &&   docker run --rm -p 8000:8000 ontime-api
FROM python:3.11-slim

# libgomp: OpenMP runtime that LightGBM needs
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY configs ./configs
COPY models ./models
COPY reports/results.json ./reports/results.json
COPY data/processed ./data/processed

ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "ontime.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
