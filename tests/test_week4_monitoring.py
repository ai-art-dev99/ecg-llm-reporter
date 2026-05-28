"""
Week 4 Monitoring Tests
========================
Tests:
  1. Metrics definitions — all counters/histograms exist
  2. Metrics recording — values update correctly
  3. /metrics endpoint — Prometheus format
  4. Timer context manager
  5. LangSmith tracer — offline (no key needed)

Run: python tests/test_week4_monitoring.py
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "

HAS_LANGSMITH = bool(os.environ.get("LANGSMITH_API_KEY"))


def run_test(name, fn, skip_reason=None):
    if skip_reason:
        print(f"{SKIP} {name}: SKIPPED — {skip_reason}")
        return None
    try:
        fn()
        print(f"{PASS} {name}")
        return True
    except AssertionError as e:
        print(f"{FAIL} {name}: {e}")
        return False
    except Exception as e:
        print(f"{FAIL} {name}: EXCEPTION — {e}")
        traceback.print_exc()
        return False


# ─── Test 1: Metrics Definitions ─────────────────────────────────────────────

def test_metrics_importable():
    from monitoring.metrics import (
        api_requests_total, api_request_duration, api_active_requests,
        ecg_processing_duration, ecg_signal_quality, ecg_heart_rate,
        ecg_anomalies_detected, ecg_rhythm_classified,
        rag_retrieval_duration, rag_chunks_retrieved, rag_top_relevance_score,
        llm_inference_duration, llm_reports_generated, llm_report_urgency,
        llm_errors_total, llm_full_pipeline_duration,
    )
    assert api_requests_total is not None
    assert llm_reports_generated is not None


def test_metrics_types():
    from prometheus_client import Counter, Histogram, Gauge
    from monitoring.metrics import (
        api_requests_total, api_request_duration,
        api_active_requests, ecg_signal_quality,
    )
    assert isinstance(api_requests_total,  Counter)
    assert isinstance(api_request_duration, Histogram)
    assert isinstance(api_active_requests, Gauge)
    assert isinstance(ecg_signal_quality,  Histogram)


# ─── Test 2: Metrics Recording ────────────────────────────────────────────────

def test_counter_increment():
    from monitoring.metrics import ecg_anomalies_detected
    before = _get_counter_value(ecg_anomalies_detected, {"anomaly_type": "tachycardia"})
    ecg_anomalies_detected.labels(anomaly_type="tachycardia").inc()
    after = _get_counter_value(ecg_anomalies_detected, {"anomaly_type": "tachycardia"})
    assert after == before + 1


def test_histogram_observe():
    from monitoring.metrics import ecg_signal_quality
    # Should not raise
    ecg_signal_quality.observe(0.85)
    ecg_signal_quality.observe(0.42)


def test_gauge_set():
    from monitoring.metrics import rag_knowledge_base_size
    rag_knowledge_base_size.set(42)
    val = rag_knowledge_base_size._value.get()
    assert val == 42.0


def test_record_ecg_metrics():
    from monitoring.metrics import record_ecg_metrics
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    loader    = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()
    record    = loader.generate_synthetic("metrics_test", heart_rate=72)
    features  = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    # Should not raise
    record_ecg_metrics(features)


def test_record_rag_metrics():
    from monitoring.metrics import record_rag_metrics
    from rag.retriever import RetrievalResult, RetrievedChunk

    chunk = RetrievedChunk(
        content="test", source="ESC", category="arrhythmia",
        relevance_score=0.82, chunk_id="test_001", doc_type="builtin",
    )
    result = RetrievalResult(query="test", chunks=[chunk], total_found=1)
    record_rag_metrics(result, duration_seconds=0.15)  # Should not raise


def test_record_llm_metrics():
    from monitoring.metrics import record_llm_metrics
    from services.report_schema import ECGReport, ReportSection, Urgency

    report = ECGReport(
        record_name="test",
        model_used="llama-3.3-70b-versatile",
        report=ReportSection(
            findings="f", interpretation="i",
            differentials="d", recommendations="r", limitations="l",
        ),
        primary_diagnosis="Normal",
        urgency=Urgency.ROUTINE,
        confidence="high",
        key_abnormalities=[],
        rag_sources_used=[],
        processing_time_ms=1000.0,
    )
    record_llm_metrics(report, duration_seconds=2.5)  # Should not raise


# ─── Test 3: Timer ────────────────────────────────────────────────────────────

def test_timer_context_manager():
    import time
    from monitoring.metrics import timer

    with timer() as t:
        time.sleep(0.05)

    assert t.elapsed >= 0.04, f"Expected ≥0.04s, got {t.elapsed:.4f}s"
    assert t.elapsed < 1.0, "Timer seems too slow"


def test_timer_captures_exception():
    from monitoring.metrics import timer

    try:
        with timer() as t:
            raise ValueError("test error")
    except ValueError:
        pass

    assert t.elapsed > 0, "Timer should still record elapsed time after exception"


# ─── Test 4: Prometheus Output ────────────────────────────────────────────────

def test_prometheus_output_format():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from monitoring.metrics import api_requests_total

    # Trigger a counter increment so metric appears in output
    api_requests_total.labels(
        endpoint="/test", method="GET", status_code="200"
    ).inc()

    output = generate_latest().decode("utf-8")
    assert "ecg_api_requests_total" in output
    assert "ecg_processing_duration_seconds" in output
    assert "llm_reports_generated_total" in output
    assert "rag_retrieval_duration_seconds" in output


# ─── Test 5: LangSmith Tracer (offline) ──────────────────────────────────────

def test_tracer_init_no_key():
    """Tracer should init without error even with no API key."""
    from monitoring.tracer import LangSmithTracer
    tracer = LangSmithTracer(api_key="")

    assert tracer.enabled is False
    assert tracer.client is None


def test_tracer_trace_noop():
    """trace_report() context manager should work even when disabled."""
    from monitoring.tracer import LangSmithTracer
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    tracer    = LangSmithTracer(api_key=None)
    loader    = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()
    record    = loader.generate_synthetic("tracer_test", heart_rate=72)
    features  = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    with tracer.trace_report(features, "llama-3.3-70b-versatile") as trace:
        assert trace.record_name == "tracer_test"
        assert trace.model == "llama-3.3-70b-versatile"
        assert trace.prompt_tokens > 0

    assert trace.duration_ms > 0
    assert trace.ended_at != ""


def test_tracer_record_result():
    from monitoring.tracer import LangSmithTracer, LLMTrace
    from rag.retriever import RetrievalResult
    from services.report_schema import ECGReport, ReportSection, Urgency

    tracer = LangSmithTracer(api_key=None)
    trace  = LLMTrace(
        run_id="test-id", record_name="test", model="test-model",
        prompt_tokens=100, started_at="2024-01-01T00:00:00",
    )
    report = ECGReport(
        record_name="test", model_used="test-model",
        report=ReportSection(
            findings="f", interpretation="i",
            differentials="d", recommendations="r", limitations="l",
        ),
        primary_diagnosis="Normal Sinus Rhythm",
        urgency=Urgency.ROUTINE, confidence="high",
        key_abnormalities=[], rag_sources_used=[], processing_time_ms=0.0,
    )
    retrieval = RetrievalResult(query="", chunks=[], total_found=3)

    tracer.record_result(trace, report, retrieval)
    assert trace.urgency   == "routine"
    assert trace.rag_chunks == 3
    assert "Normal" in trace.diagnosis


# ─── LangSmith Live Test (optional) ──────────────────────────────────────────

def test_tracer_live_connection():
    from monitoring.tracer import LangSmithTracer
    tracer = LangSmithTracer(api_key=os.environ.get("LANGSMITH_API_KEY"))
    assert tracer.enabled is True, "Expected tracer to be enabled with valid key"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_counter_value(counter, labels: dict) -> float:
    try:
        return counter.labels(**labels)._value.get()
    except Exception:
        return 0.0


# ─── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    no_ls = None if HAS_LANGSMITH else "LANGSMITH_API_KEY not set"

    tests = [
        ("Metrics: importable",              test_metrics_importable,        None),
        ("Metrics: correct types",           test_metrics_types,             None),
        ("Metrics: counter increment",       test_counter_increment,         None),
        ("Metrics: histogram observe",       test_histogram_observe,         None),
        ("Metrics: gauge set",               test_gauge_set,                 None),
        ("Metrics: record_ecg_metrics()",    test_record_ecg_metrics,        None),
        ("Metrics: record_rag_metrics()",    test_record_rag_metrics,        None),
        ("Metrics: record_llm_metrics()",    test_record_llm_metrics,        None),
        ("Timer: basic measurement",         test_timer_context_manager,     None),
        ("Timer: captures on exception",     test_timer_captures_exception,  None),
        ("Prometheus: output format",        test_prometheus_output_format,  None),
        ("Tracer: init without key",         test_tracer_init_no_key,        None),
        ("Tracer: noop context manager",     test_tracer_trace_noop,         None),
        ("Tracer: record_result()",          test_tracer_record_result,      None),
        ("Tracer: live LangSmith connect",   test_tracer_live_connection,    no_ls),
    ]

    print("\n" + "=" * 55)
    print("  MONITORING STACK — WEEK 4 TEST SUITE")
    if not HAS_LANGSMITH:
        print("  ⚠️  LANGSMITH_API_KEY not set — live test skipped")
    print("=" * 55)

    passed = skipped = failed = 0
    for name, fn, skip_reason in tests:
        result = run_test(name, fn, skip_reason)
        if result is None:  skipped += 1
        elif result:        passed  += 1
        else:               failed  += 1

    print("=" * 55)
    print(f"  Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    print("=" * 55 + "\n")

    sys.exit(0 if failed == 0 else 1)