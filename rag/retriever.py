"""
CASCADE Genuine Pinecone Clinical RAG Retriever

Production retriever operating on real vector embeddings without synthetic score inflation:
- Genuine semantic embeddings (384-dimensional).
- Explicit namespace isolation (default: clinical-guidelines-v1).
- Truthful reporting: returns 'not_configured' or 'unavailable' when Pinecone is unconfigured/offline.
- Zero fake bonuses (+0.45 removed), zero fake fallback defaults (0.95 removed).
"""

import os
from typing import Dict, List, Any, Optional
from .embeddings import SemanticEmbeddingEngine, DEFAULT_NAMESPACE, DEFAULT_VECTOR_DIM


class ClinicalRAGRetriever:
    """
    Manages semantic querying against Pinecone Cloud Vector database.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        namespace: str = DEFAULT_NAMESPACE,
        embedder: Optional[SemanticEmbeddingEngine] = None
    ):
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY", "").strip()
        self.index_name = index_name or os.environ.get("PINECONE_INDEX_NAME", "cascade-clinical-rag").strip()
        self.namespace = os.environ.get("PINECONE_NAMESPACE", namespace).strip()
        self.embedder = embedder or SemanticEmbeddingEngine()

        self.pinecone_client = None
        self.pinecone_index = None
        self.is_active = False
        self.init_error = None

        self._initialize_pinecone()

    def _initialize_pinecone(self):
        """Verifies credentials, index existence, and dimension without automatic creation during query."""
        if not self.api_key:
            self.init_error = "PINECONE_API_KEY not configured in environment"
            self.is_active = False
            return

        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.api_key)
            self.pinecone_client = pc

            # Verify index existence
            index_models = pc.list_indexes()
            existing_names = [idx.name for idx in index_models]

            if self.index_name not in existing_names:
                self.init_error = f"Index '{self.index_name}' does not exist on Pinecone account"
                self.is_active = False
                return

            # Verify dimension
            target_desc = next((idx for idx in index_models if idx.name == self.index_name), None)
            if target_desc and hasattr(target_desc, "dimension") and target_desc.dimension != DEFAULT_VECTOR_DIM:
                self.init_error = (
                    f"Index dimension mismatch: index has {target_desc.dimension}, "
                    f"embedding model requires {DEFAULT_VECTOR_DIM}"
                )
                self.is_active = False
                return

            self.pinecone_index = pc.Index(self.index_name)
            self.is_active = True
            self.init_error = None
        except Exception as e:
            self.init_error = f"Pinecone connection error: {e}"
            self.is_active = False

    def query_clinical_guidelines(
        self,
        query_text: str,
        top_k: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes semantic query against Pinecone index.
        Returns authentic matches with raw cosine similarity scores.
        """
        if not query_text or not query_text.strip():
            return {
                "status": "empty_query",
                "retrieved_chunks": [],
                "matches_count": 0,
                "message": "Query text was empty."
            }

        if not self.is_active or not self.pinecone_index:
            return {
                "status": "unavailable",
                "reason": self.init_error or "Pinecone is offline or unconfigured",
                "retrieved_chunks": [],
                "matches_count": 0,
                "message": "Live clinical guideline retrieval unavailable. Running in verified offline safe mode."
            }

        try:
            # 1. Embed query with the exact same semantic model
            query_vector = self.embedder.embed_text(query_text)

            # 2. Query Pinecone namespace
            query_kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True,
                "namespace": self.namespace
            }
            if metadata_filter:
                query_kwargs["filter"] = metadata_filter

            res = self.pinecone_index.query(**query_kwargs)

            chunks = []
            for match in res.get("matches", []):
                meta = match.get("metadata", {})
                score = match.get("score")
                raw_score = float(score) if score is not None else None

                chunks.append({
                    "chunk_id": match.get("id"),
                    "score": round(raw_score, 4) if raw_score is not None else None,
                    "title": meta.get("title", "Clinical Reference"),
                    "source_file": meta.get("filename", ""),
                    "content": meta.get("text") or meta.get("content", ""),
                    "metadata": meta
                })

            return {
                "status": "success",
                "namespace": self.namespace,
                "matches_count": len(chunks),
                "retrieved_chunks": chunks,
                "top_score": chunks[0]["score"] if chunks else None
            }

        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "retrieved_chunks": [],
                "matches_count": 0
            }
