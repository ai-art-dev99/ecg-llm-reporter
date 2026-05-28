# 🫀 ECG LLM Reporter

> AI-powered ECG analysis using RAG + LLM, fully containerised.  
> Built for the UK ML/AI job market — NHS HealthTech portfolio project.

---

## Architecture

```
ECG Input (CSV / Upload / Synthetic)
         ↓
  Signal Processing (NeuroKit2)      ← Week 1 ✅
         ↓
  Feature Extraction (HR, HRV, QT…)  ← Week 1 ✅
         ↓
  RAG Pipeline (ESC/AHA Guidelines)  ← Week 2
         ↓
  LLM Report Generation (BioMistral) ← Week 3
         ↓
  FastAPI Backend                    ← Week 1 skeleton ✅
         ↓
  Monitoring (Prometheus + Grafana)  ← Week 4
         ↓
  React/Next.js UI                   ← Week 5
         ↓
  Deploy: HuggingFace Spaces          ← Week 6
```

---

## Quick Start

```bash
# 1. Clone & configure
cp .env.example .env
# fill in HF_TOKEN in .env

# 2. Run backend only (Week 1)
docker compose up backend

# 3. Run with RAG (Week 2+)
docker compose --profile rag up

# 4. Run full stack with monitoring
docker compose --profile rag --profile monitoring --profile frontend up
```

---

## API Endpoints (Week 1)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/records` | List local ECG records |
| GET | `/records/{name}/analyze` | Analyze a named record |
| POST | `/analyze/upload` | Upload CSV ECG file |
| POST | `/analyze/synthetic` | Generate + analyze synthetic ECG |

Interactive docs: `http://localhost:8000/docs`

---

## ECG Features Extracted

| Category | Features |
|----------|----------|
| Heart Rate | Mean, Min, Max, Std, Classification |
| HRV | RMSSD, SDNN, pNN50, Mean RR |
| Intervals | PR, QRS, QT, QTc (Bazett), RR |
| Amplitudes | P, R, S, T waves, ST deviation |
| Anomalies | Tachycardia, Bradycardia, QT prolonged, Wide QRS, ST elevation/depression |

---

## Dataset

- **MIT-BIH Arrhythmia Database** — 48 half-hour ECG recordings (360 Hz, 2 leads)
- Download: [PhysioNet](https://physionet.org/content/mitdb/) (free account required)
- For development: synthetic records generated via NeuroKit2

---

## Project Roadmap

- [x] **Week 1** — ECG processing pipeline + data loader + feature extractor
- [ ] **Week 2** — RAG pipeline (ESC/AHA guidelines → ChromaDB)
- [ ] **Week 3** — LLM integration (BioMistral via HuggingFace)
- [ ] **Week 4** — Monitoring (Prometheus metrics + Grafana dashboard)
- [ ] **Week 5** — Frontend (Next.js + ECG visualiser)
- [ ] **Week 6** — Deploy to HuggingFace Spaces

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Signal Processing | NeuroKit2, SciPy, NumPy |
| Data | MIT-BIH (PhysioNet), WFDB |
| RAG | LangChain + ChromaDB |
| LLM | BioMistral-7B / Llama-3.2 |
| API | FastAPI + Pydantic |
| Monitoring | Prometheus + Grafana + LangSmith |
| Frontend | Next.js + TailwindCSS |
| Container | Docker Compose |
| Deploy | HuggingFace Spaces |
