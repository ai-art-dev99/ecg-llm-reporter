"""
LangSmith Tracer
----------------
Traces all LLM calls to LangSmith for:
  - Full prompt/response logging
  - Latency tracking
  - Error tracking
  - Run comparison across records

Gracefully disabled if LANGSMITH_API_KEY not set.
"""

import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─── Trace Data Classes ───────────────────────────────────────────────────────

@dataclass
class LLMTrace:
    run_id:        str
    record_name:   str
    model:         str
    prompt_tokens: int
    started_at:    str
    ended_at:      str = ""
    duration_ms:   float = 0.0
    urgency:       str = ""
    confidence:    str = ""
    diagnosis:     str = ""
    error:         Optional[str] = None
    rag_chunks:    int = 0
    tags:          list[str] = field(default_factory=list)


# ─── Tracer ───────────────────────────────────────────────────────────────────

class LangSmithTracer:
    """
    Wraps LangSmith tracing for ECG report generation.
    All methods are safe to call even without an API key — they just no-op.

    Usage:
        tracer = LangSmithTracer()

        with tracer.trace_report(features, "llama-3.3-70b") as trace:
            report = llm.generate_report(features, retrieval)
            tracer.record_result(trace, report, retrieval)
    """

    PROJECT = os.environ.get("LANGSMITH_PROJECT", "ecg-reporter")

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.environ.get("LANGSMITH_API_KEY")
        self.enabled = False
        self.client  = None

        if self.api_key:
            try:
                from langsmith import Client
                self.client  = Client(api_key=self.api_key)
                self.enabled = True
                print(f"[LangSmith] Tracing enabled → project: {self.PROJECT}")
            except Exception as e:
                print(f"[LangSmith] Init failed ({e}). Tracing disabled.")
        else:
            print("[LangSmith] LANGSMITH_API_KEY not set. Tracing disabled.")

    # ── Public API ────────────────────────────────────────────────────────────

    @contextmanager
    def trace_report(self, features, model: str):
        """Context manager that opens a LangSmith run and yields a trace object."""
        trace = LLMTrace(
            run_id=str(uuid.uuid4()),
            record_name=getattr(features, "record_name", "unknown"),
            model=model,
            prompt_tokens=self._estimate_tokens(features.to_prompt_context()),
            started_at=datetime.utcnow().isoformat(),
            tags=[
                f"rhythm:{features.rhythm_classification[:20]}",
                f"hr:{features.heart_rate.mean_bpm:.0f}bpm",
                f"quality:{features.signal_quality:.1f}",
            ],
        )

        run_id = None
        if self.enabled:
            run_id = self._start_run(trace, features)

        t0 = time.perf_counter()
        try:
            yield trace
            trace.duration_ms = (time.perf_counter() - t0) * 1000
            trace.ended_at    = datetime.utcnow().isoformat()
            if self.enabled and run_id:
                self._end_run(run_id, trace, error=None)
        except Exception as e:
            trace.error       = str(e)
            trace.duration_ms = (time.perf_counter() - t0) * 1000
            trace.ended_at    = datetime.utcnow().isoformat()
            if self.enabled and run_id:
                self._end_run(run_id, trace, error=str(e))
            raise

    def record_result(self, trace: LLMTrace, report, retrieval):
        """Attach report result data to an in-progress trace."""
        trace.urgency    = report.urgency.value
        trace.confidence = report.confidence
        trace.diagnosis  = report.primary_diagnosis
        trace.rag_chunks = retrieval.total_found

    def log_rag_retrieval(self, query: str, result, duration_ms: float):
        """Log a standalone RAG retrieval event."""
        if not self.enabled:
            return
        try:
            self.client.create_run(
                name="rag_retrieval",
                run_type="retriever",
                project_name=self.PROJECT,
                inputs={"query": query},
                outputs={
                    "chunks_found":      result.total_found,
                    "top_score":         result.chunks[0].relevance_score if result.chunks else 0,
                    "sources":           [c.source for c in result.chunks[:3]],
                },
                extra={"duration_ms": duration_ms},
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            print(f"[LangSmith] Failed to log RAG retrieval: {e}")

    # ── Private ───────────────────────────────────────────────────────────────

    def _start_run(self, trace: LLMTrace, features) -> Optional[str]:
        try:
            from langsmith.schemas import RunTypeEnum
            self.client.create_run(
                id=trace.run_id,
                name="ecg_report_generation",
                run_type="llm",
                project_name=self.PROJECT,
                inputs={
                    "record_name":  trace.record_name,
                    "heart_rate":   features.heart_rate.mean_bpm,
                    "rhythm":       features.rhythm_classification,
                    "anomalies":    features.anomalies.active_flags(),
                    "quality":      features.signal_quality,
                    "prompt_tokens_estimate": trace.prompt_tokens,
                },
                extra={
                    "model":  trace.model,
                    "tags":   trace.tags,
                },
                start_time=datetime.utcnow(),
            )
            return trace.run_id
        except Exception as e:
            print(f"[LangSmith] Failed to start run: {e}")
            return None

    def _end_run(self, run_id: str, trace: LLMTrace, error: Optional[str]):
        try:
            outputs = {
                "primary_diagnosis": trace.diagnosis,
                "urgency":           trace.urgency,
                "confidence":        trace.confidence,
                "rag_chunks_used":   trace.rag_chunks,
                "duration_ms":       trace.duration_ms,
            }
            self.client.update_run(
                run_id=run_id,
                outputs=outputs,
                error=error,
                end_time=datetime.utcnow(),
            )
        except Exception as e:
            print(f"[LangSmith] Failed to end run: {e}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4