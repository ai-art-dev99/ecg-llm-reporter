#!/bin/bash
set -e

echo "==> [Render] Starting ECG LLM Reporter backend..."

# ── Generate ECG data ────────────────────────────────────────────────────────
echo "==> Generating ECG data..."
python -c "
import neurokit2 as nk, pandas as pd, json, os

os.makedirs('/app/data/mitdb', exist_ok=True)

configs = {
    'normal_sinus':   (72,  0.01),
    'tachycardia':    (110, 0.01),
    'bradycardia':    (48,  0.01),
    'noisy_normal':   (75,  0.05),
}

for name, (hr, noise) in configs.items():
    csv_path  = f'/app/data/mitdb/{name}.csv'
    meta_path = f'/app/data/mitdb/{name}_meta.json'
    if os.path.exists(csv_path):
        print(f'  [skip] {name} already exists')
        continue
    ecg = nk.ecg_simulate(duration=10, sampling_rate=360, heart_rate=hr, noise=noise)
    pd.DataFrame({'MLII': ecg}).to_csv(csv_path, index=False)
    json.dump({'record_name': name, 'sampling_rate': 360, 'leads': ['MLII'], 'duration_seconds': 10}, open(meta_path, 'w'))
    print(f'  [ok] {name}')
"

# ── Ingest knowledge base ────────────────────────────────────────────────────
echo "==> Ingesting knowledge base..."
python -c "
import sys
sys.path.insert(0, '/app')

# On Render free tier: skip web scraping to save startup time
# Built-in knowledge base is always ingested
from rag.ingestor import RAGIngestor

ingestor = RAGIngestor(persist_dir='/app/chroma_db')

# Check if already ingested (persistent disk)
stats = ingestor.get_stats()
if stats['total_chunks'] > 0:
    print(f'  [skip] Already have {stats[\"total_chunks\"]} chunks in DB')
else:
    # include_web=False on Render to avoid startup timeout
    include_web = not bool(os.environ.get('RENDER', False))
    stats = ingestor.ingest_all(pdf_dir='/app/data/knowledge_base', include_web=include_web)
    print(f'  [ok] {stats[\"total\"]} chunks ingested')
" 2>/dev/null || python -c "
import sys, os
sys.path.insert(0, '/app')
from rag.ingestor import RAGIngestor
ingestor = RAGIngestor(persist_dir='/app/chroma_db')
stats = ingestor.ingest_all(pdf_dir='/app/data/knowledge_base', include_web=False)
print(f'  [ok] {stats[\"total\"]} chunks ingested')
"

echo "==> Starting API server..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"