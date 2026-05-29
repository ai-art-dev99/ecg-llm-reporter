# ECG LLM Reporter 🫀

An AI system that analyses ECG signals and generates structured clinical reports — grounded in ESC/AHA/ACC medical guidelines, not just LLM guesswork.

Upload a CSV ECG recording. The system extracts 20+ clinical features, retrieves relevant passages from established cardiology guidelines, and generates a structured diagnostic report with a full observability stack behind it.

🔗 **Live Demo:** [your-render-url]

---

## Why RAG for medical reports?

An LLM generating cardiac reports from scratch will hallucinate clinical thresholds. Grounding the pipeline in **ESC/AHA/ACC guidelines via ChromaDB** means every recommendation is traceable to a source — the same way a junior doctor checks the handbook before writing a report.

---

## Architecture

```
ECG Input (CSV upload / synthetic)
        ↓
Signal Processing — NeuroKit2
        ↓
Feature Extraction (20+ clinical features)
  HR · HRV · QT interval · ST deviation · P/R/S/T waves
        ↓
RAG Pipeline
  ESC / AHA / ACC guidelines → ChromaDB → LangChain retrieval
        ↓
LLM Report Generation
  Llama 3.3 70B via Groq API → structured JSON output
        ↓
FastAPI Backend
        ↓
Next.js Frontend — real-time ECG visualisation + report export

Observability: Prometheus · Grafana · LangSmith LLM tracing
CI/CD: GitHub Actions → Render
```

---

## Clinical Features Extracted

| Category | Features |
|---|---|
| Heart Rate | Mean · Min · Max · Std · Classification |
| HRV | RMSSD · SDNN · pNN50 · Mean RR |
| Intervals | PR · QRS · QT · QTc (Bazett) · RR |
| Amplitudes | P · R · S · T waves · ST deviation |
| Anomalies | Tachycardia · Bradycardia · QT prolongation · Wide QRS · ST elevation/depression |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Signal Processing | NeuroKit2 · SciPy · NumPy |
| RAG | LangChain · ChromaDB · ESC/AHA/ACC guidelines |
| LLM | Llama 3.3 70B · Groq API |
| Backend | Python · FastAPI · Pydantic |
| Frontend | Next.js · TypeScript · TailwindCSS |
| Observability | Prometheus · Grafana · LangSmith |
| Containerisation | Docker Compose |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## Quick Start

```bash
git clone https://github.com/ai-art-dev99/ecg-llm-reporter.git
cd ecg-llm-reporter

cp .env.example .env
# Add GROQ_API_KEY and LANGSMITH_API_KEY

# Full stack with monitoring
docker compose --profile rag --profile monitoring --profile frontend up
```

| Service | URL |
|---|---|
| Backend API | http://localhost:8000/docs |
| Frontend | http://localhost:3000 |
| Grafana | http://localhost:3001 |
| Prometheus | http://localhost:9090 |

---

## API

```bash
# Analyse an uploaded ECG file
POST /analyze/upload

# Analyse a named local record
GET /records/{name}/analyze

# Generate synthetic ECG + analyse
POST /analyze/synthetic

# Health check
GET /health
```

---

## Dataset

Uses the **MIT-BIH Arrhythmia Database** — 48 half-hour ECG recordings at 360 Hz.  
Available free at [PhysioNet](https://physionet.org/content/mitdb/).  
For development: synthetic records generated via NeuroKit2 (no download needed).

---

## Author

**Amirparsa Rouhi** · [aprouhi.com](https://aprouhi.com) · [LinkedIn](https://linkedin.com/in/amirparsa-rouhi)
