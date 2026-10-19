"""
CASCADE Pinecone Cloud & Semantic Vector Clinical RAG Engine

This module provides enterprise vector retrieval using Pinecone Cloud Vector Database
and genuine semantic vector embeddings (sentence-transformers/all-MiniLM-L6-v2)
indexing authoritative Clinical Radiology Guidelines (ACR, Fleischner, LI-RADS).

Core Principles:
- Zero fake bonuses: No artificial +0.45/+0.35 score manipulation.
- Zero fake fallbacks: No synthetic 0.95 similarity score defaults.
- Truthful status: If PINECONE_API_KEY is missing or the index is offline, returns
  truthful 'unconfigured' or 'unavailable' safe state.
- Safe import: Importing this module NEVER automatically creates indexes or upserts data.
"""

import os
from typing import Dict, Any, List, Optional
from rag.retriever import ClinicalRAGRetriever
from rag.embeddings import DEFAULT_NAMESPACE


class PineconeClinicalRAG:
    """
    Enterprise Pinecone Cloud Clinical RAG Engine backed by ClinicalRAGRetriever.
    """

    def __init__(self):
        self.api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        self.index_name = os.environ.get("PINECONE_INDEX_NAME", "cascade-clinical-rag").strip()
        self.namespace = os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE).strip()

        # Initialize retriever
        self.retriever = ClinicalRAGRetriever(
            api_key=self.api_key,
            index_name=self.index_name,
            namespace=self.namespace
        )

    @property
    def pinecone_active(self) -> bool:
        return self.retriever.is_active

    def query_dense_rag(
        self,
        organ: str = "Chest",
        severity: str = "Candidate ROI Detected",
        user_query: str = "",
        top_k: int = 2
    ) -> Dict[str, Any]:
        """
        Query Pinecone semantic index for clinical guidelines matching the patient findings.
        Returns authentic retrieved chunks, raw similarity scores, and truthful status.
        """
        # Construct search query from clinical context
        query_terms = [organ]
        if severity and severity not in ("Normal", "No Candidate ROI Flagged"):
            query_terms.append(severity)
        if user_query and user_query.strip():
            query_terms.append(user_query.strip())

        query_text = " - ".join(query_terms)

        if not self.pinecone_active:
            return {
                "kb_references": [],
                "differential_diagnosis": [],
                "top_match_condition": "Offline / Guideline Index Unconfigured",
                "similarity_score": None,
                "vector_engine": "Pinecone Cloud Index (Offline / Unconfigured)",
                "matches_count": 0,
                "retrieval_mode": "unconfigured" if not self.api_key else "unavailable",
                "retrieved_chunks": [],
                "notice": "Cloud clinical guideline retrieval is offline. No synthetic fallback scores injected."
            }

        # Query genuine Pinecone index via semantic retriever
        res = self.retriever.query_clinical_guidelines(query_text=query_text, top_k=top_k)

        if res.get("status") != "success":
            return {
                "kb_references": [],
                "differential_diagnosis": [],
                "top_match_condition": "Guideline Retrieval Error",
                "similarity_score": None,
                "vector_engine": f"Pinecone Index ({self.index_name})",
                "matches_count": 0,
                "retrieval_mode": "unavailable",
                "retrieved_chunks": [],
                "error": res.get("error") or res.get("reason", "Unknown error")
            }

        chunks = res.get("retrieved_chunks", [])
        kb_references = []
        differentials = []

        for chunk in chunks:
            title = chunk.get("title", "Clinical Reference")
            score = chunk.get("score")
            score_str = f"Raw Cosine Similarity: {score:.4f}" if score is not None else "Score: N/A"
            content_snippet = chunk.get("content", "")[:200].replace("\n", " ")
            kb_references.append(f"[{title}] {content_snippet}... ({score_str})")

            meta = chunk.get("metadata", {})
            if "differential" in meta:
                diffs = meta["differential"]
                if isinstance(diffs, list):
                    differentials.extend(diffs)
                elif isinstance(diffs, str):
                    differentials.append(diffs)

        top_match = chunks[0] if chunks else None
        top_score = top_match.get("score") if top_match else None

        return {
            "kb_references": kb_references,
            "differential_diagnosis": list(dict.fromkeys(differentials)),
            "top_match_condition": top_match.get("title") if top_match else "No Matching Guideline Found",
            "similarity_score": top_score,
            "vector_engine": f"Pinecone Cloud Index [{self.index_name} / {self.namespace}]",
            "matches_count": len(chunks),
            "retrieval_mode": "pinecone",
            "retrieved_chunks": chunks
        }


# Global Singleton Instance
pinecone_rag_engine = PineconeClinicalRAG()


if __name__ == "__main__":
    print("Testing Clean Pinecone RAG Engine...")
    result = pinecone_rag_engine.query_dense_rag(
        organ="Chest",
        severity="Candidate ROI Detected",
        user_query="Spiculated pulmonary nodule"
    )
    print("Retrieval Mode:", result["retrieval_mode"])
    print("Vector Engine:", result["vector_engine"])
    print("Similarity Score:", result["similarity_score"])
    print("Top Condition:", result["top_match_condition"])
    print("Matches Count:", result["matches_count"])
