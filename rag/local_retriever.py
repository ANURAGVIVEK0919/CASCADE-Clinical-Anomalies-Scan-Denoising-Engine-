"""
CASCADE Local Semantic Clinical RAG Retriever

Provides an authentic, offline-first clinical guideline retrieval pipeline
operating directly on the tracked Markdown corpus under knowledge_base/clinical_sources/:
- acr_appropriateness_head_trauma.md
- fleischner_society_pulmonary_nodules.md
- lirads_liver_imaging_reporting.md

Key Architecture & Contracts:
1. Reuses existing DocumentParser and DocumentChunker for deterministic parsing.
2. Embeds document chunks and queries into the same 384-dimensional space
   using sentence-transformers/all-MiniLM-L6-v2 with L2 unit normalization.
3. In-Memory Embeddings Cache: Parses and embeds the local corpus once, caching
   chunk representations in RAM to avoid expensive per-query recomputation.
4. Genuine Cosine Similarity: Uses exact dot products of normalized unit vectors
   without synthetic bonuses (+0.45), arbitrary multipliers, or fake defaults (0.95).
5. Truthful Abstention: Out-of-domain or unrelated queries below the clinical
   relevance threshold (default tau=0.38) result in an explicit abstention,
   preventing hallucinated or irrelevant clinical guidance.
6. Safe Error Propagation: If embedding dependencies are missing, reports a clear
   offline dependency error without silently fabricating results.
7. Lazy Initialization: Does not load embedding weights at import/startup until
   retrieval is actually invoked.
"""

import os
import threading
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from .parsers import DocumentParser
from .chunker import DocumentChunker, DocumentChunk
from .embeddings import SemanticEmbeddingEngine, DEFAULT_VECTOR_DIM

# Determine absolute repository root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLINICAL_SOURCES_DIR = os.path.join(REPO_ROOT, "knowledge_base", "clinical_sources")

# Calibrated clinical relevance threshold (calibrated to reject OOD while admitting relevant queries)
DEFAULT_RELEVANCE_THRESHOLD = 0.38

# Guideline differential reference mapping (authoritative topics from tracked documents)
DOCUMENT_DIFFERENTIAL_MAP = {
    "acr_appropriateness_head_trauma.md": [
        "Epidural Hematoma (EDH)",
        "Subdural Hematoma (SDH)",
        "Subarachnoid Hemorrhage (SAH)",
        "Intraparenchymal Contusion / Diffuse Axonal Injury (DAI)"
    ],
    "fleischner_society_pulmonary_nodules.md": [
        "Solid Pulmonary Nodule (<6mm, 6-8mm, >8mm)",
        "Pure Ground-Glass Nodule (GGN)",
        "Part-Solid Pulmonary Nodule",
        "Incidental Pulmonary Nodule (Benign vs Malignant Consideration)"
    ],
    "lirads_liver_imaging_reporting.md": [
        "Hepatocellular Carcinoma (HCC)",
        "LR-4 (Probable HCC) / LR-5 (Definite HCC)",
        "Hepatic Hemangioma",
        "Hepatic Steatosis (Fatty Liver)",
        "Focal Nodular Hyperplasia (FNH)"
    ]
}


class LocalClinicalRAGRetriever:
    """
    In-memory semantic vector retriever operating on tracked local clinical guidelines.
    Caches parsed document chunks and 384-dimensional embeddings in RAM.
    """

    # Class-level shared cache: {sources_dir: {"chunks": [...], "vectors": np.ndarray}}
    _shared_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self,
        sources_dir: Optional[str] = None,
        relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD,
        embedder: Optional[SemanticEmbeddingEngine] = None
    ):
        self.sources_dir = os.path.abspath(sources_dir or DEFAULT_CLINICAL_SOURCES_DIR)
        self.relevance_threshold = float(relevance_threshold)
        self.embedder = embedder or SemanticEmbeddingEngine()
        self.dependency_error: Optional[str] = None

    def _ensure_corpus_indexed(self) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Loads, chunks, and embeds the local guideline documents if not already in cache.
        Thread-safe in-memory cache shared across retriever instances.
        """
        with self._cache_lock:
            if self.sources_dir in self._shared_cache:
                cached = self._shared_cache[self.sources_dir]
                return cached["chunks"], cached["vectors"]

            if not os.path.exists(self.sources_dir):
                return [], None

            # 1. Parse all markdown documents in the sources directory
            parser = DocumentParser()
            chunker = DocumentChunker(chunk_size=600, chunk_overlap=120)

            corpus_chunks: List[Dict[str, Any]] = []
            chunk_texts: List[str] = []

            md_files = sorted([
                f for f in os.listdir(self.sources_dir)
                if f.lower().endswith(".md")
            ])

            for fname in md_files:
                fpath = os.path.join(self.sources_dir, fname)
                parsed = parser.parse_file(fpath)
                if not parsed.get("success"):
                    continue

                chunks = chunker.chunk_document(parsed)
                for chunk in chunks:
                    chunk_dict = {
                        "chunk_id": chunk.chunk_id,
                        "source_file": chunk.source_file,
                        "title": chunk.title,
                        "content": chunk.content,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata
                    }
                    corpus_chunks.append(chunk_dict)
                    chunk_texts.append(chunk.content)

            if not corpus_chunks:
                self._shared_cache[self.sources_dir] = {"chunks": [], "vectors": None}
                return [], None

            # 2. Embed all chunks in batches
            try:
                embeddings = self.embedder.embed_batch(chunk_texts, batch_size=32)
                vectors = np.array(embeddings, dtype=np.float32)
                self._shared_cache[self.sources_dir] = {
                    "chunks": corpus_chunks,
                    "vectors": vectors
                }
                return corpus_chunks, vectors
            except Exception as e:
                self.dependency_error = str(e)
                raise RuntimeError(f"Local semantic embedding failed: {e}")

    def query_clinical_guidelines(
        self,
        query_text: str,
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Executes semantic query against the local in-memory guideline vector corpus.
        Returns authentic matched chunks with source attribution and computed cosine similarities.
        """
        if not query_text or not query_text.strip():
            return {
                "status": "empty_query",
                "retrieval_mode": "empty_query",
                "retrieved_chunks": [],
                "matches_count": 0,
                "top_score": None,
                "top_match_condition": "Empty Query",
                "kb_references": [],
                "differential_diagnosis": [],
                "message": "Query text was empty."
            }

        # Verify or index corpus
        try:
            chunks, vectors = self._ensure_corpus_indexed()
        except Exception as e:
            return {
                "status": "error",
                "retrieval_mode": "dependency_error",
                "error": f"Offline RAG dependency error: {e}",
                "retrieved_chunks": [],
                "matches_count": 0,
                "top_score": None,
                "top_match_condition": "Offline RAG Dependency Error",
                "kb_references": [],
                "differential_diagnosis": [],
                "notice": "Offline semantic guideline retrieval requires sentence-transformers."
            }

        if not chunks or vectors is None or len(vectors) == 0:
            return {
                "status": "unavailable",
                "retrieval_mode": "unavailable",
                "retrieved_chunks": [],
                "matches_count": 0,
                "top_score": None,
                "top_match_condition": "No Clinical Sources Available",
                "kb_references": [],
                "differential_diagnosis": [],
                "message": f"No valid guideline documents found in {self.sources_dir}."
            }

        # Embed query text
        try:
            query_vec = np.array(self.embedder.embed_text(query_text), dtype=np.float32)
        except Exception as e:
            return {
                "status": "error",
                "retrieval_mode": "dependency_error",
                "error": f"Failed embedding query: {e}",
                "retrieved_chunks": [],
                "matches_count": 0,
                "top_score": None,
                "top_match_condition": "Query Embedding Error",
                "kb_references": [],
                "differential_diagnosis": []
            }

        # Compute cosine similarity: since vectors are L2 normalized unit vectors,
        # dot product is mathematically identical to cosine similarity.
        similarities = np.dot(vectors, query_vec)

        # Rank chunks by similarity descending
        ranked_indices = np.argsort(-similarities)
        top_idx = ranked_indices[0]
        top_score = float(similarities[top_idx])

        # Truthful abstention gate: if highest score is below clinical threshold, abstain
        if top_score < self.relevance_threshold:
            return {
                "status": "abstained",
                "retrieval_mode": "abstained",
                "matches_count": 0,
                "top_score": round(top_score, 4),
                "threshold": self.relevance_threshold,
                "top_match_condition": "No Relevant Clinical Guideline Found",
                "kb_references": [],
                "differential_diagnosis": [],
                "retrieved_chunks": [],
                "notice": (
                    f"Local semantic retrieval abstained: top similarity ({top_score:.4f}) "
                    f"below clinical relevance threshold ({self.relevance_threshold:.4f})."
                )
            }

        # Filter top_k matches that meet or exceed threshold
        retrieved_chunks = []
        differentials = []
        kb_references = []

        for idx in ranked_indices[:top_k]:
            sim = float(similarities[idx])
            if sim < self.relevance_threshold:
                continue

            chunk = chunks[idx]
            raw_score = round(sim, 4)
            title = chunk.get("title", "Clinical Reference")
            source_file = chunk.get("source_file", "")
            content = chunk.get("content", "")
            snippet = content[:200].replace("\n", " ").strip()

            retrieved_chunk = {
                "chunk_id": chunk.get("chunk_id"),
                "score": raw_score,
                "title": title,
                "source_file": source_file,
                "content": content,
                "snippet": snippet,
                "metadata": chunk.get("metadata", {})
            }
            retrieved_chunks.append(retrieved_chunk)

            # Format human-readable citation with source and exact similarity
            kb_references.append(
                f"[{title}] {snippet}... (Source: {source_file} | Raw Cosine Similarity: {raw_score:.4f})"
            )

            # Collect source-grounded differential considerations
            if source_file in DOCUMENT_DIFFERENTIAL_MAP:
                differentials.extend(DOCUMENT_DIFFERENTIAL_MAP[source_file])

        top_match = retrieved_chunks[0] if retrieved_chunks else None

        return {
            "status": "success",
            "retrieval_mode": "local_semantic",
            "matches_count": len(retrieved_chunks),
            "top_score": top_match["score"] if top_match else None,
            "threshold": self.relevance_threshold,
            "top_match_condition": top_match.get("title") if top_match else "Guideline Retrieved",
            "kb_references": kb_references,
            "differential_diagnosis": list(dict.fromkeys(differentials)),
            "retrieved_chunks": retrieved_chunks,
            "vector_engine": f"Local Semantic RAG (all-MiniLM-L6-v2 / 384-d in-memory)"
        }
