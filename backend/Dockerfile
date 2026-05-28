FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache sentence-transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('Model cached')" || echo "Model cache skipped (offline)"

COPY . .

RUN mkdir -p /app/data/mitdb /app/data/knowledge_base /app/chroma_db
RUN chmod +x /app/start.sh

EXPOSE ${PORT:-8000}

CMD ["/bin/bash", "/app/start.sh"]