"""
Week 2 RAG Pipeline Tests
==========================
Tests:
  1. Knowledge base — content and categories
  2. TF-IDF embedding (offline fallback)
  3. Ingestor — built-in docs loading
  4. Retriever — semantic search
  5. retrieve_for_features() — ECG-aware queries
  6. format_for_prompt() — LLM context output

Run: python tests/test_week2_rag.py
"""

import sys
import os
import shutil

# Add backend directory to path (same fix as week 1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import traceback
import tempfile

PASS = "✅"
FAIL = "❌"

TEST_DB_DIR = tempfile.mkdtemp(prefix="ecg_test_chroma_")


def run_test(name, fn):
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


# ─── Test 1: Knowledge Base ───────────────────────────────────────────────────

def test_kb_has_documents():
    from rag.knowledge_base import get_all_documents
    docs = get_all_documents()
    assert len(docs) >= 10, f"Expected ≥10 docs, got {len(docs)}"

def test_kb_categories():
    from rag.knowledge_base import get_categories, get_by_category
    cats = get_categories()
    assert "arrhythmia" in cats
    assert "ischemia" in cats
    assert "intervals" in cats
    arrhythmia_docs = get_by_category("arrhythmia")
    assert len(arrhythmia_docs) >= 2

def test_kb_document_structure():
    from rag.knowledge_base import get_all_documents
    for doc in get_all_documents():
        assert "id" in doc
        assert "content" in doc
        assert "category" in doc
        assert "source" in doc
        assert len(doc["content"]) > 50

# ─── Test 2: Embeddings ───────────────────────────────────────────────────────

def test_tfidf_embedding():
    from rag.ingestor import TFIDFEmbedding
    ef = TFIDFEmbedding()
    texts = ["normal sinus rhythm 72 bpm", "tachycardia fast heart rate 110"]
    ef.fit(texts)
    vecs = ef(texts)
    assert len(vecs) == 2
    assert len(vecs[0]) == TFIDFEmbedding.DIM
    # Vectors should be different
    import numpy as np
    assert not np.allclose(vecs[0], vecs[1])

def test_tfidf_similarity():
    from rag.ingestor import TFIDFEmbedding
    import numpy as np
    ef = TFIDFEmbedding()
    texts = [
        "normal sinus rhythm ECG heart rate 72 bpm normal",
        "normal sinus rhythm ECG heart rate 70 bpm regular",
        "ventricular fibrillation chaotic rhythm defibrillation",
    ]
    ef.fit(texts)
    vecs = np.array(ef(texts))
    sim_similar = np.dot(vecs[0], vecs[1])
    sim_different = np.dot(vecs[0], vecs[2])
    assert sim_similar > sim_different, \
        f"Similar texts should have higher cosine sim: {sim_similar:.3f} vs {sim_different:.3f}"

# ─── Test 3: Ingestor ─────────────────────────────────────────────────────────

def test_ingestor_builtin():
    from rag.ingestor import RAGIngestor
    ingestor = RAGIngestor(persist_dir=TEST_DB_DIR)
    n = ingestor._ingest_builtin()
    assert n > 0, "No built-in docs were ingested"
    stats = ingestor.get_stats()
    assert stats["total_chunks"] > 0

def test_ingestor_no_duplicates():
    from rag.ingestor import RAGIngestor
    ingestor = RAGIngestor(persist_dir=TEST_DB_DIR)
    count_before = ingestor.collection.count()
    # Run again — should not add duplicates
    n = ingestor._ingest_builtin()
    count_after = ingestor.collection.count()
    assert count_after == count_before, \
        f"Duplicate ingestion! Before: {count_before}, After: {count_after}"
    assert n == 0, f"Expected 0 new docs on re-ingest, got {n}"

def test_ingest_all_no_web():
    from rag.ingestor import RAGIngestor
    ingestor = RAGIngestor(persist_dir=TEST_DB_DIR)
    stats = ingestor.ingest_all(pdf_dir=None, include_web=False)
    assert stats["total"] > 0
    assert stats["web"] == 0

# ─── Test 4: Retriever ────────────────────────────────────────────────────────

def test_retriever_basic_query():
    from rag.retriever import ECGRetriever
    r = ECGRetriever(persist_dir=TEST_DB_DIR)
    result = r.retrieve("tachycardia heart rate fast", n_results=3)
    assert result.total_found > 0, "Expected at least 1 result"
    assert len(result.chunks) > 0
    assert result.chunks[0].relevance_score > 0

def test_retriever_afib_query():
    from rag.retriever import ECGRetriever
    r = ECGRetriever(persist_dir=TEST_DB_DIR)
    result = r.retrieve("atrial fibrillation irregular rhythm no P waves", n_results=3)
    assert result.total_found > 0
    contents = " ".join([c.content for c in result.chunks]).lower()
    assert any(word in contents for word in ["atrial", "fibrillation", "irregular", "rhythm"])

def test_retriever_qt_query():
    from rag.retriever import ECGRetriever
    r = ECGRetriever(persist_dir=TEST_DB_DIR)
    result = r.retrieve("prolonged QT interval Torsades de Pointes risk", n_results=3)
    assert result.total_found > 0
    contents = " ".join([c.content for c in result.chunks]).lower()
    assert any(word in contents for word in ["qt", "torsades", "prolonged"])

def test_retriever_min_relevance():
    from rag.retriever import ECGRetriever
    r = ECGRetriever(persist_dir=TEST_DB_DIR)
    result_loose = r.retrieve("ECG heart", n_results=5, min_relevance=0.0)
    result_strict = r.retrieve("ECG heart", n_results=5, min_relevance=0.9)
    assert result_loose.total_found >= result_strict.total_found

# ─── Test 5: retrieve_for_features() ─────────────────────────────────────────

def test_retrieve_for_normal_features():
    from rag.retriever import ECGRetriever
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    loader = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()
    retriever = ECGRetriever(persist_dir=TEST_DB_DIR)

    record = loader.generate_synthetic("test_retrieve", heart_rate=72)
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    result = retriever.retrieve_for_features(features, n_results=5)
    assert result.total_found > 0, "retrieve_for_features returned no results"
    print(f"       Normal: {result.total_found} chunks, top score: {result.chunks[0].relevance_score:.3f}")

def test_retrieve_for_tachycardia_features():
    from rag.retriever import ECGRetriever
    from services.data_loader import ECGDataLoader
    from services.ecg_processor import ECGFeatureExtractor

    loader = ECGDataLoader(data_dir="data/mitdb")
    extractor = ECGFeatureExtractor()
    retriever = ECGRetriever(persist_dir=TEST_DB_DIR)

    record = loader.generate_synthetic("test_tachy_rag", heart_rate=120, rhythm="tachycardia")
    features = extractor.extract(record.signal, record.sampling_rate, record.record_name)

    result = retriever.retrieve_for_features(features)
    assert result.total_found > 0
    contents = " ".join([c.content for c in result.chunks]).lower()
    assert any(w in contents for w in ["tachycardia", "heart rate", "rate"]), \
        "Expected tachycardia-related content in results"

# ─── Test 6: Prompt Formatting ────────────────────────────────────────────────

def test_format_for_prompt():
    from rag.retriever import ECGRetriever
    r = ECGRetriever(persist_dir=TEST_DB_DIR)
    result = r.retrieve("normal sinus rhythm ECG", n_results=3)
    prompt_ctx = result.format_for_prompt()
    assert "RETRIEVED CLINICAL GUIDELINES" in prompt_ctx
    assert len(prompt_ctx) > 100
    print(f"\n--- Prompt Context Sample (first 300 chars) ---")
    print(prompt_ctx[:300])

def test_format_empty_result():
    from rag.retriever import RetrievalResult
    empty = RetrievalResult(query="test", chunks=[], total_found=0)
    text = empty.format_for_prompt()
    assert "No relevant guidelines" in text

# ─── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("Knowledge base: has documents", test_kb_has_documents),
        ("Knowledge base: categories correct", test_kb_categories),
        ("Knowledge base: document structure", test_kb_document_structure),
        ("TF-IDF: embedding dimensions", test_tfidf_embedding),
        ("TF-IDF: similar texts score higher", test_tfidf_similarity),
        ("Ingestor: built-in docs loaded", test_ingestor_builtin),
        ("Ingestor: no duplicates on re-run", test_ingestor_no_duplicates),
        ("Ingestor: ingest_all (no web)", test_ingest_all_no_web),
        ("Retriever: basic query", test_retriever_basic_query),
        ("Retriever: AFib query", test_retriever_afib_query),
        ("Retriever: QT query", test_retriever_qt_query),
        ("Retriever: min_relevance filter", test_retriever_min_relevance),
        ("Retriever: retrieve_for_features (normal)", test_retrieve_for_normal_features),
        ("Retriever: retrieve_for_features (tachycardia)", test_retrieve_for_tachycardia_features),
        ("Prompt: format_for_prompt()", test_format_for_prompt),
        ("Prompt: empty result handling", test_format_empty_result),
    ]

    print("\n" + "=" * 55)
    print("  ECG RAG PIPELINE — WEEK 2 TEST SUITE")
    print("=" * 55)

    passed = sum(run_test(name, fn) for name, fn in tests)
    total = len(tests)

    print("=" * 55)
    print(f"  Result: {passed}/{total} tests passed")
    if passed < total:
        print(f"  {total - passed} test(s) failed — see above")
    print("=" * 55 + "\n")

    # Cleanup temp DB
    shutil.rmtree(TEST_DB_DIR, ignore_errors=True)
    sys.exit(0 if passed == total else 1)