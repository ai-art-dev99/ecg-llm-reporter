"""
RAG Ingestor
------------
Loads documents from three sources into ChromaDB:
  1. Built-in knowledge base (always available)
  2. PDF files from data/knowledge_base/ (user-provided)
  3. Medical websites (Wikipedia, PubMed — when internet available)

Embedding strategy:
  - Docker (internet): sentence-transformers/all-MiniLM-L6-v2 via HuggingFace
  - Offline fallback: TF-IDF based custom embeddings (no downloads needed)
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb import EmbeddingFunction, Documents, Embeddings

from rag.knowledge_base import get_all_documents

# ─── Embedding Functions ──────────────────────────────────────────────────────

class SentenceTransformerEmbedding(EmbeddingFunction):
    """
    Uses sentence-transformers/all-MiniLM-L6-v2.
    384-dim, fast, high quality for medical text.
    Requires internet on first use (model download ~90MB).
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        print(f"[Embedding] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("[Embedding] Model loaded.")

    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(list(input), show_progress_bar=False).tolist()


class TFIDFEmbedding(EmbeddingFunction):
    """
    TF-IDF based embedding — works completely offline, no downloads.
    Lower quality than transformer models but functional.
    Auto-used when sentence-transformers unavailable.
    """
    DIM = 512

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=self.DIM,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self._fitted = False
        self._corpus: list[str] = []

    def fit(self, texts: list[str]):
        self._corpus = texts
        self.vectorizer.fit(texts)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        if not self._fitted:
            self.vectorizer.fit(texts)
            self._fitted = True
        mat = self.vectorizer.transform(texts).toarray()
        # Pad to fixed DIM
        result = np.zeros((len(texts), self.DIM))
        w = min(mat.shape[1], self.DIM)
        result[:, :w] = mat[:, :w]
        # L2 normalise
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (result / norms).tolist()


def build_embedding_function() -> EmbeddingFunction:
    """Try sentence-transformers first, fall back to TF-IDF."""
    try:
        ef = SentenceTransformerEmbedding()
        print("[Embedding] Using sentence-transformers (high quality)")
        return ef
    except Exception as e:
        print(f"[Embedding] sentence-transformers unavailable ({e}). Using TF-IDF fallback.")
        return TFIDFEmbedding()


# ─── Document Chunker ─────────────────────────────────────────────────────────

class TextChunker:
    """Split long text into overlapping chunks for better retrieval."""

    def __init__(self, chunk_size: int = 400, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> list[str]:
        # Clean whitespace
        text = re.sub(r"\s+", " ", text.strip())

        if len(text) <= self.chunk_size:
            return [text]

        # Split on sentence boundaries when possible
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) <= self.chunk_size:
                current += " " + sentence if current else sentence
            else:
                if current:
                    chunks.append(current.strip())
                # Start new chunk with overlap
                overlap_text = current[-self.overlap:] if len(current) > self.overlap else current
                current = overlap_text + " " + sentence

        if current:
            chunks.append(current.strip())

        return [c for c in chunks if len(c) > 50]


# ─── Source Loaders ───────────────────────────────────────────────────────────

class PDFLoader:
    """Load and extract text from PDF files."""

    @staticmethod
    def load(pdf_path: str) -> list[dict]:
        """Returns list of {content, metadata} dicts."""
        try:
            from pypdf import PdfReader
        except ImportError:
            print("[PDF] pypdf not installed. Skipping PDF loading.")
            return []

        try:
            reader = PdfReader(pdf_path)
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and len(text.strip()) > 100:
                    pages.append({
                        "content": text.strip(),
                        "metadata": {
                            "source": Path(pdf_path).name,
                            "page": i + 1,
                            "type": "pdf",
                        },
                    })
            print(f"[PDF] Loaded {len(pages)} pages from {Path(pdf_path).name}")
            return pages
        except Exception as e:
            print(f"[PDF] Failed to load {pdf_path}: {e}")
            return []


class WebLoader:
    """Scrape medical content from Wikipedia and PubMed."""

    ECG_URLS = [
        {
            "url": "https://en.wikipedia.org/wiki/Electrocardiography",
            "title": "Electrocardiography — Wikipedia",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Arrhythmia",
            "title": "Cardiac Arrhythmia — Wikipedia",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Atrial_fibrillation",
            "title": "Atrial Fibrillation — Wikipedia",
        },
        {
            "url": "https://en.wikipedia.org/wiki/QT_interval",
            "title": "QT Interval — Wikipedia",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Bundle_branch_block",
            "title": "Bundle Branch Block — Wikipedia",
        },
        {
            "url": "https://en.wikipedia.org/wiki/Myocardial_infarction",
            "title": "Myocardial Infarction — Wikipedia",
        },
    ]

    HEADERS = {
        "User-Agent": "ECG-LLM-Reporter/1.0 (medical-education; contact@example.com)"
    }

    @classmethod
    def load_all(cls, timeout: int = 10) -> list[dict]:
        """Load all configured medical URLs. Skips on failure."""
        import urllib.request
        from bs4 import BeautifulSoup

        documents = []
        for entry in cls.ECG_URLS:
            try:
                req = urllib.request.Request(entry["url"], headers=cls.HEADERS)
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    soup = BeautifulSoup(response.read(), "html.parser")

                    # Remove nav, footer, references
                    for tag in soup(["nav", "footer", "table", "script", "style", ".reflist"]):
                        tag.decompose()

                    # Extract paragraphs
                    paragraphs = [
                        p.get_text().strip()
                        for p in soup.find_all("p")
                        if len(p.get_text().strip()) > 100
                    ]

                    if paragraphs:
                        documents.append({
                            "content": "\n\n".join(paragraphs[:20]),  # First 20 paragraphs
                            "metadata": {
                                "source": entry["title"],
                                "url": entry["url"],
                                "type": "web",
                            },
                        })
                        print(f"[Web] Loaded: {entry['title']}")
                    time.sleep(0.5)  # Be polite

            except Exception as e:
                print(f"[Web] Skipped {entry['title']}: {e}")

        return documents


# ─── Main Ingestor ────────────────────────────────────────────────────────────

class RAGIngestor:
    """
    Orchestrates loading from all sources and inserting into ChromaDB.

    Usage:
        ingestor = RAGIngestor(persist_dir="./chroma_db")
        ingestor.ingest_all(pdf_dir="data/knowledge_base", include_web=True)
    """

    COLLECTION_NAME = "ecg_knowledge"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.chunker = TextChunker(chunk_size=400, overlap=80)
        self.ef = build_embedding_function()

        # Persistent ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[ChromaDB] Collection '{self.COLLECTION_NAME}' ready. "
              f"Current docs: {self.collection.count()}")

    def ingest_all(
        self,
        pdf_dir: Optional[str] = None,
        include_web: bool = True,
        force_reload: bool = False,
    ) -> dict:
        """
        Load all sources into ChromaDB.
        Returns summary of what was ingested.
        """
        if force_reload:
            print("[Ingestor] Force reload — clearing collection...")
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": "cosine"},
            )

        stats = {"builtin": 0, "pdf": 0, "web": 0, "total": 0}

        # 1. Built-in knowledge base (always)
        print("\n[Ingestor] Loading built-in knowledge base...")
        n = self._ingest_builtin()
        stats["builtin"] = n

        # 2. PDFs
        if pdf_dir and Path(pdf_dir).exists():
            print(f"\n[Ingestor] Scanning PDFs in {pdf_dir}...")
            pdfs = list(Path(pdf_dir).glob("*.pdf"))
            print(f"[Ingestor] Found {len(pdfs)} PDF(s)")
            for pdf in pdfs:
                pages = PDFLoader.load(str(pdf))
                n = self._ingest_documents(pages, source_type="pdf")
                stats["pdf"] += n
        else:
            print(f"\n[Ingestor] No PDF directory found — skipping PDF ingestion")
            print(f"           (Add PDFs to '{pdf_dir}' and re-run to include them)")

        # 3. Web
        if include_web:
            print("\n[Ingestor] Loading medical websites...")
            web_docs = WebLoader.load_all()
            n = self._ingest_documents(web_docs, source_type="web")
            stats["web"] = n

        stats["total"] = self.collection.count()
        print(f"\n[Ingestor] ✅ Done. Total chunks in DB: {stats['total']}")
        print(f"           Built-in: {stats['builtin']} | PDF: {stats['pdf']} | Web: {stats['web']}")
        return stats

    def _ingest_builtin(self) -> int:
        """Ingest the built-in ECG knowledge base."""
        docs = get_all_documents()
        added = 0

        for doc in docs:
            chunks = self.chunker.split(doc["content"])
            for i, chunk in enumerate(chunks):
                doc_id = f"{doc['id']}_chunk{i}"
                # Skip if already exists
                existing = self.collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue
                self.collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": doc["source"],
                        "category": doc["category"],
                        "title": doc["title"],
                        "type": "builtin",
                        "chunk_index": i,
                    }],
                )
                added += 1

        print(f"[Ingestor] Built-in: added {added} new chunks")
        return added

    def _ingest_documents(self, docs: list[dict], source_type: str) -> int:
        """Ingest a list of {content, metadata} dicts."""
        added = 0

        for doc in docs:
            chunks = self.chunker.split(doc["content"])
            meta = doc.get("metadata", {})
            source_name = meta.get("source", "unknown")

            for i, chunk in enumerate(chunks):
                # Deterministic ID based on content hash
                content_hash = hashlib.md5(chunk.encode()).hexdigest()[:10]
                doc_id = f"{source_type}_{content_hash}_{i}"

                existing = self.collection.get(ids=[doc_id])
                if existing["ids"]:
                    continue

                self.collection.add(
                    ids=[doc_id],
                    documents=[chunk],
                    metadatas=[{
                        "source": source_name,
                        "type": source_type,
                        "chunk_index": i,
                        **{k: str(v) for k, v in meta.items()},
                    }],
                )
                added += 1

        return added

    def get_stats(self) -> dict:
        """Return collection statistics."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection": self.COLLECTION_NAME,
            "persist_dir": self.persist_dir,
        }