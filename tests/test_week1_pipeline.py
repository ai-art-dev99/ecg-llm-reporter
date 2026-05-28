"""
Week 1 Pipeline Test
====================
Tests:
  1. DataLoader — synthetic + CSV
  2. ECGFeatureExtractor — all record types
  3. to_prompt_context() — LLM-ready output
  4. Anomaly detection on known pathological patterns

Run: python tests/test_week1_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import traceback

from services.data_loader import ECGDataLoader
from services.ecg_processor import ECGFeatureExtractor


PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


def run_test(name: str, fn):
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


# ─── Setup ────────────────────────────────────────────────────────────────────

loader = ECGDataLoader(data_dir="data/mitdb")
extractor = ECGFeatureExtractor()


# ─── Test 1: DataLoader ───────────────────────────────────────────────────────

def test_loader_synthetic_normal():
    record = loader.generate_synthetic("test_normal", heart_rate=72, rhythm="normal")
    assert record.source == "synthetic"
    assert record.sampling_rate == 360
    assert len(record.signal) == 360 * 10   # sampling_rate × duration
    assert "MLII" in record.leads


def test_loader_synthetic_tachycardia():
    record = loader.generate_synthetic("test_tachy", rhythm="tachycardia")
    assert record.metadata["target_heart_rate"] >= 100


def test_loader_synthetic_bradycardia():
    record = loader.generate_synthetic("test_brady", rhythm="bradycardia")
    assert record.metadata["target_heart_rate"] <= 60


def test_loader_from_array():
    signal = np.sin(np.linspace(0, 20 * np.pi, 3600))
    record = loader.load_from_array(signal, sampling_rate=360, record_name="sine_test")
    assert record.source == "upload"
    assert len(record.signal) == 3600


def test_loader_list_records():
    records = loader.list_available_records()
    assert isinstance(records, list)
    print(f"       Found {len(records)} records: {records}")


def test_loader_csv_files_exist():
    records = loader.list_available_records()
    assert "normal_sinus" in records, f"normal_sinus not found. Got: {records}"
    assert "tachycardia" in records


def test_loader_load_record():
    record = loader.load_record("normal_sinus")
    assert record.signal is not None
    assert len(record.signal) > 0


# ─── Test 2: Feature Extractor ────────────────────────────────────────────────

def test_extractor_normal():
    record = loader.generate_synthetic("normal_72", heart_rate=72)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    assert features.heart_rate.mean_bpm > 0
    assert 60 <= features.heart_rate.mean_bpm <= 90, \
        f"Expected normal HR ~72, got {features.heart_rate.mean_bpm}"
    assert features.heart_rate.classification == "normal"
    assert features.r_peak_count > 5


def test_extractor_tachycardia_detection():
    record = loader.generate_synthetic("tachy_120", heart_rate=120, rhythm="tachycardia")
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    assert features.heart_rate.mean_bpm > 90, \
        f"Expected HR >90, got {features.heart_rate.mean_bpm}"
    assert features.anomalies.tachycardia, "Tachycardia flag not set"


def test_extractor_bradycardia_detection():
    record = loader.generate_synthetic("brady_45", heart_rate=45, rhythm="bradycardia")
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    assert features.heart_rate.mean_bpm < 65, \
        f"Expected HR <65, got {features.heart_rate.mean_bpm}"
    assert features.anomalies.bradycardia, "Bradycardia flag not set"


def test_extractor_hrv():
    record = loader.generate_synthetic("hrv_test", heart_rate=72, duration=30)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    assert features.hrv.rmssd_ms is not None
    assert features.hrv.sdnn_ms is not None
    assert features.hrv.mean_rr_ms is not None
    assert 500 < features.hrv.mean_rr_ms < 1200, \
        f"Mean RR should be ~833ms for 72bpm, got {features.hrv.mean_rr_ms}"


def test_extractor_intervals():
    record = loader.generate_synthetic("intervals_test", heart_rate=72)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    # At least some intervals should be detected
    has_any = any([
        features.intervals.rr_interval_ms,
        features.intervals.pr_interval_ms,
        features.intervals.qrs_duration_ms,
    ])
    assert has_any, "No intervals could be extracted"


def test_extractor_signal_quality():
    # Good signal
    record = loader.generate_synthetic("clean", noise=0.005)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)
    assert features.signal_quality > 0.3

    # Noisy signal
    record_noisy = loader.generate_synthetic("noisy", noise=0.5)
    features_noisy = extractor.extract(
        record_noisy.signal, record_noisy.sampling_rate, record_noisy.record_name
    )
    # Just check it doesn't crash
    assert features_noisy is not None


# ─── Test 3: Prompt Context ───────────────────────────────────────────────────

def test_prompt_context_format():
    record = loader.generate_synthetic("prompt_test", heart_rate=72)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    ctx = features.to_prompt_context()
    assert "Heart Rate" in ctx or "HEART RATE" in ctx
    assert "bpm" in ctx
    assert str(record.record_name) in ctx
    assert "ANOMALY FLAGS" in ctx
    print(f"\n--- Prompt Context Sample ---\n{ctx[:400]}...\n")


def test_prompt_context_has_summary():
    record = loader.load_record("tachycardia")
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)
    ctx = features.to_prompt_context()
    assert "SUMMARY:" in ctx


# ─── Test 4: End-to-End on All Saved Records ─────────────────────────────────

def test_pipeline_all_records():
    records = loader.list_available_records()
    for r_name in records:
        record = loader.load_record(r_name)
        features = extractor.extract(record.signal, record.sampling_rate, record.record_name)
        assert features is not None
        assert features.summary
        print(f"       {r_name}: HR={features.heart_rate.mean_bpm:.1f} | {features.rhythm_classification}")


# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("DataLoader: synthetic normal", test_loader_synthetic_normal),
        ("DataLoader: synthetic tachycardia", test_loader_synthetic_tachycardia),
        ("DataLoader: synthetic bradycardia", test_loader_synthetic_bradycardia),
        ("DataLoader: from numpy array", test_loader_from_array),
        ("DataLoader: list records", test_loader_list_records),
        ("DataLoader: CSV records exist", test_loader_csv_files_exist),
        ("DataLoader: load_record() CSV", test_loader_load_record),
        ("Extractor: normal sinus rhythm", test_extractor_normal),
        ("Extractor: tachycardia detection", test_extractor_tachycardia_detection),
        ("Extractor: bradycardia detection", test_extractor_bradycardia_detection),
        ("Extractor: HRV metrics", test_extractor_hrv),
        ("Extractor: wave intervals", test_extractor_intervals),
        ("Extractor: signal quality", test_extractor_signal_quality),
        ("Prompt context: format check", test_prompt_context_format),
        ("Prompt context: summary present", test_prompt_context_has_summary),
        ("E2E: all saved records", test_pipeline_all_records),
    ]

    print("\n" + "=" * 55)
    print("  ECG PIPELINE — WEEK 1 TEST SUITE")
    print("=" * 55)

    passed = sum(run_test(name, fn) for name, fn in tests)
    total = len(tests)

    print("=" * 55)
    print(f"  Result: {passed}/{total} tests passed")
    if passed < total:
        print(f"  {total - passed} tests failed — see above for details")
    print("=" * 55 + "\n")

    sys.exit(0 if passed == total else 1)
