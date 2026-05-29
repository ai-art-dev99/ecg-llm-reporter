"""
ECG LLM Reporter — FastAPI Backend
Week 1: ECG processing
Week 2: RAG pipeline
Week 3: LLM report generation
Week 4: Prometheus metrics + LangSmith tracing
"""

import io
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from services.data_loader import ECGDataLoader
from services.ecg_processor import ECGFeatureExtractor
from services.report_schema import ECGReport
from monitoring.metrics import (
    api_requests_total, api_request_duration, api_active_requests,
    api_errors_total, app_startup_timestamp, app_info,
    rag_knowledge_base_size, record_ecg_metrics, record_rag_metrics,
    record_llm_metrics, timer, llm_errors_total, llm_full_pipeline_duration,
    ecg_records_processed, ecg_processing_duration,
)

# ─── Global services ─────────────────────────────────────────────────────────

loader    = ECGDataLoader(data_dir="data/mitdb")
extractor = ECGFeatureExtractor()
retriever = None
llm       = None
tracer    = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, llm, tracer

    app_startup_timestamp.set_to_current_time()
    app_info.labels(version="0.4.0", model="llama-3.3-70b-versatile").set(1)

    # ── RAG ──────────────────────────────────────────────────────────────────
    try:
        from rag.ingestor  import RAGIngestor
        from rag.retriever import ECGRetriever
        ingestor = RAGIngestor(persist_dir="./chroma_db")
        ingestor.ingest_all(pdf_dir="data/knowledge_base", include_web=True)
        # Pass the same client so in-memory data is shared
        retriever = ECGRetriever(persist_dir="./chroma_db", client=ingestor.client)
        stats = retriever.get_collection_stats()
        rag_knowledge_base_size.set(stats["total_chunks"])
        print(f"[Startup] RAG ready — {stats['total_chunks']} chunks")
    except Exception as e:
        print(f"[Startup] RAG init failed: {e}")

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        try:
            from services.llm_service import LLMService
            llm = LLMService(api_key=groq_key)
            print("[Startup] LLM ready")
        except Exception as e:
            print(f"[Startup] LLM init failed: {e}")

    # ── LangSmith tracer ─────────────────────────────────────────────────────
    try:
        from monitoring.tracer import LangSmithTracer
        tracer = LangSmithTracer()
    except Exception as e:
        print(f"[Startup] Tracer init failed: {e}")

    yield


# ─── App Setup ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ECG LLM Reporter API",
    description="ECG signal processing + LLM-generated clinical reports",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Prometheus Middleware ────────────────────────────────────────────────────

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Record request metrics for every endpoint."""
    # Skip metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    api_active_requests.inc()
    t0 = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        api_errors_total.labels(
            endpoint=request.url.path,
            error_type=type(e).__name__,
        ).inc()
        raise
    finally:
        duration = time.perf_counter() - t0
        endpoint = request.url.path
        api_active_requests.dec()
        api_requests_total.labels(
            endpoint=endpoint,
            method=request.method,
            status_code=str(status_code),
        ).inc()
        api_request_duration.labels(endpoint=endpoint).observe(duration)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class ECGAnalysisResponse(BaseModel):
    record_name:               str
    sampling_rate:             int
    duration_seconds:          float
    signal_quality:            float
    heart_rate_mean:           float
    heart_rate_classification: str
    rhythm:                    str
    r_peak_count:              int
    anomalies:                 list[str]
    summary:                   str
    prompt_context:            str
    processing_notes:          list[str]
    processing_time_ms:        float


class SyntheticRequest(BaseModel):
    heart_rate: int   = 72
    duration:   int   = 10
    rhythm:     str   = "normal"
    noise:      float = 0.01


class SyntheticReportRequest(BaseModel):
    heart_rate:  int   = 72
    duration:    int   = 10
    rhythm:      str   = "normal"
    noise:       float = 0.01
    include_rag: bool  = True


# ─── System Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "version": "0.4.0",
        "rag":     retriever is not None,
        "llm":     llm is not None,
        "tracing": tracer.enabled if tracer else False,
    }


@app.get("/status")
def status():
    rag_stats = retriever.get_collection_stats() if retriever else {"ready": False}
    return {
        "rag":      rag_stats,
        "llm":      {"ready": llm is not None, "model": "llama-3.3-70b-versatile"},
        "tracing":  {"langsmith": tracer.enabled if tracer else False},
        "data":     {"records": loader.list_available_records()},
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics scrape endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ─── ECG Analysis Endpoints ──────────────────────────────────────────────────

@app.get("/records")
def list_records():
    return {"records": loader.list_available_records()}


@app.get("/records/{record_name}/analyze", response_model=ECGAnalysisResponse)
def analyze_record(record_name: str):
    try:
        record = loader.load_record(record_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _run_extraction(record, source="mitdb")


@app.post("/analyze/upload", response_model=ECGAnalysisResponse)
async def analyze_upload(file: UploadFile = File(...), sampling_rate: int = 360):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files supported.")
    contents = await file.read()
    try:
        df  = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        cols = df.select_dtypes(include=[np.number]).columns
        if not len(cols):
            raise ValueError("No numeric columns.")
        signal = df[cols[0]].dropna().values
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")
    record = loader.load_from_array(signal, sampling_rate, record_name=file.filename)
    return _run_extraction(record, source="upload")


@app.post("/analyze/synthetic", response_model=ECGAnalysisResponse)
def analyze_synthetic(req: SyntheticRequest):
    record = loader.generate_synthetic(
        record_name=f"synthetic_{req.rhythm}_{req.heart_rate}bpm",
        heart_rate=req.heart_rate, duration=req.duration,
        rhythm=req.rhythm, noise=req.noise,
    )
    return _run_extraction(record, source="synthetic")


# ─── Report Endpoints ────────────────────────────────────────────────────────

@app.get("/records/{record_name}/report")
def report_from_record(record_name: str, include_rag: bool = True):
    _require_llm()
    try:
        record = loader.load_record(record_name)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _run_full_pipeline(record, include_rag)


@app.post("/analyze/synthetic/report")
def report_from_synthetic(req: SyntheticReportRequest):
    _require_llm()
    record = loader.generate_synthetic(
        record_name=f"synthetic_{req.rhythm}_{req.heart_rate}bpm",
        heart_rate=req.heart_rate, duration=req.duration,
        rhythm=req.rhythm, noise=req.noise,
    )
    return _run_full_pipeline(record, req.include_rag)


@app.post("/analyze/upload/report")
async def report_from_upload(file: UploadFile = File(...), sampling_rate: int = 360):
    _require_llm()
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files supported.")
    contents = await file.read()
    try:
        df   = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        cols = df.select_dtypes(include=[np.number]).columns
        signal = df[cols[0]].dropna().values
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {e}")
    record = loader.load_from_array(signal, sampling_rate, record_name=file.filename)
    return _run_full_pipeline(record, include_rag=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_extraction(record, source: str = "unknown") -> ECGAnalysisResponse:
    with timer() as t:
        features = extractor.extract(
            record.signal, record.sampling_rate, record.record_name
        )

    # Record metrics
    ecg_processing_duration.observe(t.elapsed)
    ecg_records_processed.labels(source=source).inc()
    record_ecg_metrics(features)

    return ECGAnalysisResponse(
        record_name=features.record_name,
        sampling_rate=features.sampling_rate,
        duration_seconds=features.duration_seconds,
        signal_quality=features.signal_quality,
        heart_rate_mean=features.heart_rate.mean_bpm,
        heart_rate_classification=features.heart_rate.classification,
        rhythm=features.rhythm_classification,
        r_peak_count=features.r_peak_count,
        anomalies=features.anomalies.active_flags(),
        summary=features.summary,
        prompt_context=features.to_prompt_context(),
        processing_notes=features.processing_notes,
        processing_time_ms=round(t.elapsed * 1000, 2),
    )


def _run_full_pipeline(record, include_rag: bool = True) -> dict:
    from rag.retriever import RetrievalResult

    pipeline_start = time.perf_counter()

    # ── Step 1: ECG features ─────────────────────────────────────────────────
    with timer() as ecg_t:
        features = extractor.extract(
            record.signal, record.sampling_rate, record.record_name
        )
    ecg_processing_duration.observe(ecg_t.elapsed)
    record_ecg_metrics(features)

    # ── Step 2: RAG retrieval ────────────────────────────────────────────────
    with timer() as rag_t:
        if include_rag and retriever:
            retrieval = retriever.retrieve_for_features(features, n_results=6)
        else:
            retrieval = RetrievalResult(query="", chunks=[], total_found=0)
    record_rag_metrics(retrieval, rag_t.elapsed)

    if tracer:
        tracer.log_rag_retrieval(
            query=retrieval.query,
            result=retrieval,
            duration_ms=rag_t.elapsed * 1000,
        )

    # ── Step 3: LLM generation ───────────────────────────────────────────────
    with timer() as llm_t:
        try:
            if tracer:
                with tracer.trace_report(features, llm.MODEL) as trace:
                    report: ECGReport = llm.generate_report(features, retrieval)
                    tracer.record_result(trace, report, retrieval)
            else:
                report: ECGReport = llm.generate_report(features, retrieval)
        except Exception as e:
            llm_errors_total.labels(error_type=type(e).__name__).inc()
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

    record_llm_metrics(report, llm_t.elapsed)
    llm_full_pipeline_duration.observe(time.perf_counter() - pipeline_start)

    return {
        "report":   report.model_dump(),
        "text":     report.to_text(),
        "features": {
            "heart_rate": features.heart_rate.mean_bpm,
            "rhythm":     features.rhythm_classification,
            "anomalies":  features.anomalies.active_flags(),
            "quality":    features.signal_quality,
        },
        "rag": {
            "chunks_retrieved": retrieval.total_found,
            "sources":          list({c.source for c in retrieval.chunks}),
        },
        "timing_ms": {
            "ecg":      round(ecg_t.elapsed * 1000, 2),
            "rag":      round(rag_t.elapsed * 1000, 2),
            "llm":      round(llm_t.elapsed * 1000, 2),
            "total":    round((time.perf_counter() - pipeline_start) * 1000, 2),
        },
    }


def _require_llm():
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not available. Set GROQ_API_KEY in .env. "
                   "Free key at: https://console.groq.com",
        )