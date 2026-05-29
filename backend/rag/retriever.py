"""
RAG Retriever
-------------
Queries ChromaDB to find relevant clinical guidelines
for a given set of ECG features.

Supports:
  - Semantic search via ChromaDB
  - Query building from ECGFeatures object
  - Category filtering (e.g., only 'ischemia' docs)
  - Relevance scoring and deduplication
"""

from dataclasses import dataclass
from typing import Optional
import os

import chromadb

from rag.ingestor import RAGIngestor, build_embedding_function


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    content: str
    source: str
    category: str
    relevance_score: float      # 0–1 (1 = most similar)
    chunk_id: str
    doc_type: str               # builtin | pdf | web


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    total_found: int

    def format_for_prompt(self, max_chars: int = 3000) -> str:
        """Format retrieved chunks as context block for LLM prompt."""
        if not self.chunks:
            return "No relevant guidelines retrieved."

        lines = ["RETRIEVED CLINICAL GUIDELINES", "=" * 40]
        total = 0

        for i, chunk in enumerate(self.chunks, 1):
            header = f"\n[{i}] {chunk.source} (relevance: {chunk.relevance_score:.2f})"
            body = chunk.content.strip()

            if total + len(header) + len(body) > max_chars:
                lines.append(f"\n[...{len(self.chunks) - i + 1} more results truncated]")
                break

            lines.append(header)
            lines.append(body)
            lines.append("-" * 30)
            total += len(header) + len(body)

        return "\n".join(lines)


# ─── Retriever ────────────────────────────────────────────────────────────────

class ECGRetriever:
    """
    Retrieves relevant ECG guidelines from ChromaDB.

    Usage:
        retriever = ECGRetriever(persist_dir="./chroma_db")
        result = retriever.retrieve_for_features(features)
        print(result.format_for_prompt())
    """

    COLLECTION_NAME = "ecg_knowledge"

    def __init__(self, persist_dir: str = "./chroma_db", client=None):
        self.persist_dir = persist_dir
        self.ef = build_embedding_function()

        if client:
            # Use shared client (important for in-memory mode)
            self.client = client
        else:
            use_memory = os.environ.get("RENDER") == "true"
            if use_memory:
                self.client = chromadb.EphemeralClient()
            else:
                self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        category_filter: Optional[str] = None,
        min_relevance: float = 0.0,
    ) -> RetrievalResult:
        """
        Core retrieval method.

        Args:
            query: natural language query
            n_results: max chunks to return
            category_filter: limit to specific category (e.g., 'arrhythmia')
            min_relevance: filter out chunks below this score
        """
        if self.collection.count() == 0:
            return RetrievalResult(query=query, chunks=[], total_found=0)

        where = {"category": category_filter} if category_filter else None

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[Retriever] Query failed: {e}")
            return RetrievalResult(query=query, chunks=[], total_found=0)

        chunks = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0–1
            score = max(0.0, 1.0 - (dist / 2.0))

            if score < min_relevance:
                continue

            chunks.append(RetrievedChunk(
                content=doc,
                source=meta.get("source", "Unknown"),
                category=meta.get("category", "general"),
                relevance_score=round(score, 3),
                chunk_id=results["ids"][0][len(chunks)],
                doc_type=meta.get("type", "unknown"),
            ))

        return RetrievalResult(query=query, chunks=chunks, total_found=len(chunks))

    def retrieve_for_features(self, features, n_results: int = 8) -> RetrievalResult:
        """
        Build smart queries from ECGFeatures and retrieve relevant guidelines.
        Runs multiple targeted queries and merges results.
        """
        queries = self._build_queries_from_features(features)
        all_chunks: dict[str, RetrievedChunk] = {}

        # Run each query and collect unique chunks
        for query in queries:
            result = self.retrieve(query, n_results=3, min_relevance=0.2)
            for chunk in result.chunks:
                # Deduplicate by chunk_id, keep highest score
                if chunk.chunk_id not in all_chunks or \
                   chunk.relevance_score > all_chunks[chunk.chunk_id].relevance_score:
                    all_chunks[chunk.chunk_id] = chunk

        # Sort by relevance
        sorted_chunks = sorted(all_chunks.values(), key=lambda x: x.relevance_score, reverse=True)
        top_chunks = sorted_chunks[:n_results]

        # Build combined query for display
        combined_query = " | ".join(queries[:2])
        return RetrievalResult(
            query=combined_query,
            chunks=top_chunks,
            total_found=len(top_chunks),
        )

    def _build_queries_from_features(self, features) -> list[str]:
        """
        Generate targeted queries based on active ECG findings.
        Always includes a baseline query.
        """
        queries = []
        hr = features.heart_rate
        intervals = features.intervals
        anomalies = features.anomalies

        # 1. Rhythm-based query
        queries.append(f"{features.rhythm_classification} ECG diagnosis management")

        # 2. HR classification
        if anomalies.tachycardia:
            queries.append("sinus tachycardia causes treatment ECG")
        elif anomalies.bradycardia:
            queries.append("sinus bradycardia causes pacemaker ECG")
        else:
            queries.append(f"normal sinus rhythm heart rate {hr.mean_bpm:.0f} bpm")

        # 3. QT interval
        if anomalies.qt_prolonged:
            queries.append("prolonged QTc Torsades de Pointes risk management")
        elif anomalies.qt_short:
            queries.append("short QT interval syndrome risk arrhythmia")

        # 4. QRS
        if anomalies.wide_qrs:
            queries.append("wide QRS bundle branch block LBBB RBBB ECG interpretation")

        # 5. ST changes
        if anomalies.st_elevation:
            queries.append("ST elevation STEMI myocardial infarction ECG treatment PCI")
        elif anomalies.st_depression:
            queries.append("ST depression NSTEMI ischemia unstable angina")

        # 6. PR interval
        if anomalies.pr_prolonged:
            queries.append("first degree AV block prolonged PR interval causes")

        # 7. HRV
        if features.hrv.rmssd_ms and features.hrv.rmssd_ms < 20:
            queries.append("reduced heart rate variability HRV clinical significance")

        return queries

    def get_collection_stats(self) -> dict:
        """Return stats about the current knowledge base."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection": self.COLLECTION_NAME,
            "persist_dir": self.persist_dir,
            "ready": count > 0,
        }