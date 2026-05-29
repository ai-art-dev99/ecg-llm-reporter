FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# sentence-transformers only for local Docker (not Render)
# Installed separately so Render build doesn't OOM
RUN if [ "$RENDER" != "true" ]; then \
    pip install --no-cache-dir sentence-transformers==3.0.0 && \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" || true; \
    fi

COPY . .

RUN mkdir -p /app/data/mitdb /app/data/knowledge_base /app/chroma_db
RUN chmod +x /app/start.sh

EXPOSE ${PORT:-8000}

CMD ["/bin/bash", "/app/start.sh"]