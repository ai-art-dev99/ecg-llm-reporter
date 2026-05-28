"""
Metrics Module
--------------
All Prometheus metrics for the ECG LLM Reporter.

Categories:
  - API metrics      (requests, latency, errors)
  - ECG metrics      (processing time, signal quality, anomalies)
  - RAG metrics      (retrieval time, chunks found, relevance scores)
  - LLM metrics      (inference time, token usage, report urgency)
  - System metrics   (active requests, startup time)
"""

from prometheus_client import Counter, Histogram, Gauge, Summary, REGISTRY
from prometheus_client import start_http_server
import time

# ─── API Metrics ──────────────────────────────────────────────────────────────

api_requests_total = Counter(
    "ecg_api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_code"],
)

api_request_duration = Histogram(
    "ecg_api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

api_active_requests = Gauge(
    "ecg_api_active_requests",
    "Number of currently active API requests",
)

api_errors_total = Counter(
    "ecg_api_errors_total",
    "Total API errors",
    ["endpoint", "error_type"],
)

# ─── ECG Processing Metrics ───────────────────────────────────────────────────

ecg_processing_duration = Histogram(
    "ecg_processing_duration_seconds",
    "ECG feature extraction duration",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

ecg_signal_quality = Histogram(
    "ecg_signal_quality_score",
    "ECG signal quality score (0-1)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ecg_heart_rate = Histogram(
    "ecg_heart_rate_bpm",
    "Detected heart rate in BPM",
    buckets=[30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 200],
)

ecg_anomalies_detected = Counter(
    "ecg_anomalies_detected_total",
    "Total anomaly flags detected",
    ["anomaly_type"],
)

ecg_rhythm_classified = Counter(
    "ecg_rhythm_classified_total",
    "ECG rhythm classifications",
    ["rhythm_type"],
)

ecg_records_processed = Counter(
    "ecg_records_processed_total",
    "Total ECG records processed",
    ["source"],   # mitdb | synthetic | upload
)

# ─── RAG Metrics ──────────────────────────────────────────────────────────────

rag_retrieval_duration = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)

rag_chunks_retrieved = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks retrieved per query",
    buckets=[0, 1, 2, 3, 4, 5, 6, 8, 10],
)

rag_top_relevance_score = Histogram(
    "rag_top_relevance_score",
    "Top relevance score from RAG retrieval (0-1)",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

rag_knowledge_base_size = Gauge(
    "rag_knowledge_base_chunks_total",
    "Total chunks in ChromaDB knowledge base",
)

# ─── LLM Metrics ─────────────────────────────────────────────────────────────

llm_inference_duration = Histogram(
    "llm_inference_duration_seconds",
    "LLM inference duration (full round-trip to Groq)",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0],
)

llm_reports_generated = Counter(
    "llm_reports_generated_total",
    "Total ECG reports generated",
    ["urgency", "confidence"],
)

llm_report_urgency = Counter(
    "llm_report_urgency_total",
    "Report urgency distribution",
    ["urgency"],   # routine | urgent | emergency
)

llm_errors_total = Counter(
    "llm_errors_total",
    "LLM generation errors",
    ["error_type"],   # api_error | parse_error | timeout
)

llm_full_pipeline_duration = Histogram(
    "llm_full_pipeline_duration_seconds",
    "Full pipeline: ECG → RAG → LLM → Report",
    buckets=[1, 2, 5, 10, 20, 30, 60],
)

# ─── System Metrics ───────────────────────────────────────────────────────────

app_startup_timestamp = Gauge(
    "ecg_app_startup_timestamp",
    "Unix timestamp when the app started",
)

app_info = Gauge(
    "ecg_app_info",
    "Application version info",
    ["version", "model"],
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def record_ecg_metrics(features):
    """Record all ECG feature metrics from an ECGFeatures object."""
    ecg_signal_quality.observe(features.signal_quality)
    ecg_heart_rate.observe(features.heart_rate.mean_bpm)
    ecg_records_processed.labels(source="unknown").inc()

    for flag in features.anomalies.active_flags():
        ecg_anomalies_detected.labels(anomaly_type=flag).inc()

    # Simplify rhythm to a short label for cardinality control
    rhythm = features.rhythm_classification.lower()
    if "normal" in rhythm:
        label = "normal_sinus"
    elif "tachycardia" in rhythm:
        label = "tachycardia"
    elif "bradycardia" in rhythm:
        label = "bradycardia"
    elif "fibrillation" in rhythm or "irregular" in rhythm:
        label = "irregular"
    else:
        label = "other"

    ecg_rhythm_classified.labels(rhythm_type=label).inc()


def record_rag_metrics(retrieval_result, duration_seconds: float):
    """Record RAG retrieval metrics."""
    rag_retrieval_duration.observe(duration_seconds)
    rag_chunks_retrieved.observe(retrieval_result.total_found)

    if retrieval_result.chunks:
        top_score = retrieval_result.chunks[0].relevance_score
        rag_top_relevance_score.observe(top_score)


def record_llm_metrics(report, duration_seconds: float):
    """Record LLM report generation metrics."""
    llm_inference_duration.observe(duration_seconds)
    llm_reports_generated.labels(
        urgency=report.urgency.value,
        confidence=report.confidence,
    ).inc()
    llm_report_urgency.labels(urgency=report.urgency.value).inc()


class timer:
    """Context manager for timing code blocks."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start