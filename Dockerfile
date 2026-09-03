# Ships the prebuilt index and both ONNX models, so the container needs no
# network at runtime and a cold start is not a model download for whoever
# arrives first. Build the database with `make build` before building this.
FROM python:3.11-slim

# PORT is read at runtime so the same image runs anywhere: Hugging Face Spaces
# expects 7860, Fly is configured for 8000, Render injects its own $PORT.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    HOME=/app \
    FASTEMBED_CACHE_PATH=/app/models

# Hugging Face Spaces runs containers as uid 1000 and the process must be able
# to write its own cache, so build as that user rather than root.
RUN useradd -m -u 1000 app && mkdir -p /app/models && chown -R app:app /app
WORKDIR /app
USER app
ENV PATH="/app/.local/bin:${PATH}"

COPY --chown=app:app pyproject.toml README.md ./
COPY --chown=app:app src/ ./src/
RUN pip install --no-cache-dir --user -e .

# Bake the models in. Downloading them on first request turns a cold start into
# a 30-second stall for whoever happens to arrive first.
RUN python -c "\
from fastembed import TextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

COPY --chown=app:app web/ ./web/
COPY --chown=app:app evaldata/ ./evaldata/
COPY --chown=app:app data/cfr.db ./data/cfr.db

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD \
    python -c "import os,httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health').json()['ok'] else 1)"

# One worker on purpose: the vector matrix and the rate-limit buckets are both
# per-process, so a second worker doubles memory and silently doubles the
# effective rate limit. Shell form so $PORT expands at runtime.
CMD uvicorn cfr.api:app --host 0.0.0.0 --port ${PORT} --workers 1
