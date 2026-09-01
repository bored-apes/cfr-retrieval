# The image ships the prebuilt index, so the container starts in seconds and
# needs no network at runtime. Build the database first with `make build`.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FASTEMBED_CACHE_PATH=/app/models

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e .

# Bake the ONNX models in rather than downloading on first request, so a cold
# start is not a 30 second stall for whoever hits it first.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY web/ ./web/
COPY evaldata/ ./evaldata/
COPY data/cfr.db ./data/cfr.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/api/health').json()['ok'] else 1)"

# One worker: the rate limiter and the vector matrix are both per-process, so a
# second worker doubles the memory and silently doubles the effective rate limit.
CMD ["uvicorn", "cfr.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
