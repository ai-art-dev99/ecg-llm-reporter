"""
Week 3 LLM Integration Tests
=============================
Tests:
  1. Report schema — structure and serialisation
  2. Prompt builder — correct context injection
  3. LLM service — with real Groq API (skipped if no key)
  4. Full pipeline — features → RAG → report

Run: python tests/test_week3_llm.py
Set GROQ_API_KEY env var to enable live API tests.
"""

import os
import sys
import shutil
import tempfile
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = "✅"
FAIL = "❌"
SKIP = "⏭️ "

GROQ_KEY      = os.environ.get("GROQ_API_KEY")
HAS_GROQ      = bool(GROQ_KEY)
TEST_DB_DIR   = tempfile.mkdtemp(prefix="ecg_test_w3_")


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


# ─── Test 1: Report Schema ────────────────────────────────────────────────────

def test_report_schema_creation():
    from services.report_schema import ECGReport, ReportSection, Urgency
    report = ECGReport(
        record_name="test_001",
        model_used="llama-3.3-70b-versatile",
        report=ReportSection(
            findings="Heart rate 110 bpm. Regular rhythm.",
            interpretation="Sinus tachycardia.",
            differentials="SVT, anxiety, fever.",
            recommendations="Treat underlying cause.",
            limitations="AI-generated, requires clinical correlation.",
        ),
        primary_diagnosis="Sinus Tachycardia",
        urgency=Urgency.ROUTINE,
        confidence="high",
        key_abnormalities=["Tachycardia (110 bpm)"],
        rag_sources_used=["ESC Guidelines 2019"],
        processing_time_ms=1234.5,
    )
    assert report.record_name == "test_001"
    assert report.urgency == Urgency.ROUTINE
    assert len(report.key_abnormalities) == 1


def test_report_to_text():
    from services.report_schema import ECGReport, ReportSection, Urgency
    report = ECGReport(
        record_name="test_txt",
        model_used="llama-3.3-70b-versatile",
        report=ReportSection(
            findings="HR 72 bpm.", interpretation="Normal.",
            differentials="None.", recommendations="Routine follow-up.",
            limitations="AI only.",
        ),
        primary_diagnosis="Normal Sinus Rhythm",
        urgency=Urgency.ROUTINE,
        confidence="high",
        key_abnormalities=[],
        rag_sources_used=[],
        processing_time_ms=500.0,
    )
    text = report.to_text()
    assert "ECG ANALYSIS REPORT" in text
    assert "Normal Sinus Rhythm" in text
    assert "FINDINGS" in text
    assert "RECOMMENDATIONS" in text


def test_report_serialisation():
    from services.report_schema import ECGReport, ReportSection, Urgency
    report = ECGReport(
        record_name="serial_test",
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
        processing_time_ms=0.0,
    )
    d = report.model_dump()
    assert d["urgency"] == "routine"
    assert "report" in d
    assert "findings" in d["report"]


def test_urgency_enum():
    from services.report_schema import Urgency
    assert Urgency("routine")   == Urgency.ROUTINE
    assert Urgency("urgent")    == Urgency.URGENT
    assert Urgency("emergency") == Urgency.EMERGENCY


# ─── Test 2: Prompt Builder ───────────────────────────────────────────────────

def test_prompt_contains_ecg_data():
    from services.llm_service import build_user_prompt
    ecg_ctx = "Heart Rate: 110 bpm\nRhythm: Sinus Tachycardia"
    rag_ctx = "ESC Guidelines: tachycardia management..."
    prompt = build_user_prompt(ecg_ctx, rag_ctx)
    assert "110 bpm" in prompt
    assert "ESC Guidelines" in prompt
    assert "JSON" in prompt or "json" in prompt.lower()


def test_system_prompt_has_format():
    from services.llm_service import SYSTEM_PROMPT
    assert "primary_diagnosis" in SYSTEM_PROMPT
    assert "urgency" in SYSTEM_PROMPT
    assert "findings" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT


# ─── Test 3: LLM Service (requires GROQ_API_KEY) ─────────────────────────────

def test_llm_init():
    from services.llm_service import LLMService
    svc = LLMService(api_key=GROQ_KEY)
    assert svc.client is not None


def test_llm_generate_simple():
    from services.llm_service import LLMService
    from rag.retriever import RetrievalResult
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    svc      = LLMService(api_key=GROQ_KEY)
    loader   = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()

    record   = loader.generate_synthetic("llm_test_normal", heart_rate=72)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)
    empty_rag = RetrievalResult(query="", chunks=[], total_found=0)

    report = svc.generate_report(features, empty_rag)

    assert report.primary_diagnosis
    assert report.urgency is not None
    assert report.report.findings
    assert report.processing_time_ms > 0
    print(f"       Diagnosis: {report.primary_diagnosis}")
    print(f"       Urgency: {report.urgency.value} | Confidence: {report.confidence}")


def test_llm_tachycardia_flagged_urgent():
    from services.llm_service import LLMService
    from rag.retriever import RetrievalResult
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor
    from services.report_schema import Urgency

    svc       = LLMService(api_key=GROQ_KEY)
    loader    = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()

    record   = loader.generate_synthetic("llm_test_tachy", heart_rate=140, rhythm="tachycardia")
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)
    empty_rag = RetrievalResult(query="", chunks=[], total_found=0)

    report = svc.generate_report(features, empty_rag)

    assert report.primary_diagnosis
    # 140 bpm should trigger at least urgent
    assert report.urgency in [Urgency.URGENT, Urgency.EMERGENCY], \
        f"Expected urgent/emergency for 140 bpm, got: {report.urgency}"
    print(f"       Tachycardia diagnosis: {report.primary_diagnosis}")


# ─── Test 4: Full Pipeline ────────────────────────────────────────────────────

def test_full_pipeline_with_rag():
    from services.llm_service import LLMService
    from rag.ingestor  import RAGIngestor
    from rag.retriever import ECGRetriever
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    # Init RAG
    ingestor = RAGIngestor(persist_dir=TEST_DB_DIR)
    ingestor.ingest_all(pdf_dir=None, include_web=False)
    retriever = ECGRetriever(persist_dir=TEST_DB_DIR)

    # Init LLM
    svc       = LLMService(api_key=GROQ_KEY)
    loader    = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()

    # Run
    record    = loader.generate_synthetic("full_pipeline_test", heart_rate=72)
    features  = extractor.extract(record.signal, record.sampling_rate, record.record_name)
    retrieval = retriever.retrieve_for_features(features, n_results=5)
    report    = svc.generate_report(features, retrieval)

    assert report.primary_diagnosis
    assert report.report.findings
    assert report.processing_time_ms > 0

    # Some RAG sources should be referenced
    print(f"       RAG sources: {report.rag_sources_used}")
    print(f"       Report length: {len(report.to_text())} chars")
    print(f"       Processing time: {report.processing_time_ms:.0f}ms")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    no_key = None if HAS_GROQ else "GROQ_API_KEY not set"

    tests = [
        ("Schema: report creation",      test_report_schema_creation,       None),
        ("Schema: to_text() format",      test_report_to_text,               None),
        ("Schema: serialisation",         test_report_serialisation,         None),
        ("Schema: urgency enum",          test_urgency_enum,                 None),
        ("Prompt: contains ECG data",     test_prompt_contains_ecg_data,     None),
        ("Prompt: system prompt format",  test_system_prompt_has_format,     None),
        ("LLM: init Groq client",         test_llm_init,                     no_key),
        ("LLM: generate normal report",   test_llm_generate_simple,          no_key),
        ("LLM: tachycardia flagged",      test_llm_tachycardia_flagged_urgent, no_key),
        ("Pipeline: full RAG + LLM",      test_full_pipeline_with_rag,       no_key),
    ]

    print("\n" + "=" * 55)
    print("  ECG LLM SERVICE — WEEK 3 TEST SUITE")
    if not HAS_GROQ:
        print("  ⚠️  GROQ_API_KEY not set — API tests will be skipped")
    print("=" * 55)

    passed = skipped = failed = 0
    for name, fn, skip_reason in tests:
        result = run_test(name, fn, skip_reason)
        if result is None:  skipped += 1
        elif result:        passed += 1
        else:               failed += 1

    print("=" * 55)
    print(f"  Passed: {passed} | Skipped: {skipped} | Failed: {failed}")
    if not HAS_GROQ:
        print("  💡 To run all tests: set GROQ_API_KEY and re-run")
    print("=" * 55 + "\n")

    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)
    sys.exit(0 if failed == 0 else 1)