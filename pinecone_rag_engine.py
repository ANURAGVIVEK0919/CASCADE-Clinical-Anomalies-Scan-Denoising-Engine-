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
from rag.local_retriever import LocalClinicalRAGRetriever
from rag.embeddings import DEFAULT_NAMESPACE


class PineconeClinicalRAG:
    """
    Enterprise Pinecone Cloud Clinical RAG Engine with Local Semantic RAG Fallback.
    """

    def __init__(self, local_retriever: Optional[LocalClinicalRAGRetriever] = None):
        self.api_key = os.environ.get("PINECONE_API_KEY", "").strip()
        self.index_name = os.environ.get("PINECONE_INDEX_NAME", "cascade-clinical-rag").strip()
        self.namespace = os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE).strip()

        # Initialize Pinecone cloud retriever
        self.retriever = ClinicalRAGRetriever(
            api_key=self.api_key,
            index_name=self.index_name,
            namespace=self.namespace
        )
        # Initialize local semantic retriever (lazy by default)
        self.local_retriever = local_retriever or LocalClinicalRAGRetriever()

    @property
    def pinecone_active(self) -> bool:
        return self.retriever.is_active

    def query_dense_rag(
        self,
        organ: str = "Chest",
        severity: str = "Candidate ROI Detected",
        user_query: str = "",
        top_k: int = 2,
        allow_local_fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Query Pinecone semantic index for clinical guidelines matching patient findings.
        If Pinecone is unconfigured or unavailable and allow_local_fallback is True,
        seamlessly queries the local tracked clinical guideline corpus using genuine
        semantic sentence embeddings (384-d, L2 normalized).
        """
        # Construct search query from clinical context: prioritize user's direct query
        if user_query and user_query.strip():
            query_text = user_query.strip()
        else:
            query_terms = [organ]
            if severity and severity not in ("Normal", "No Candidate ROI Flagged"):
                query_terms.append(severity)
            query_text = " - ".join(query_terms)

        # --------------------------------------------------------------------
        # Priority 1: Genuine Pinecone Cloud Retrieval (when active & working)
        # --------------------------------------------------------------------
        if self.pinecone_active:
            res = self.retriever.query_clinical_guidelines(query_text=query_text, top_k=top_k)
            if res.get("status") == "success":
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

        # --------------------------------------------------------------------
        # Priority 2: Local Semantic RAG Fallback (on tracked local guidelines)
        # --------------------------------------------------------------------
        if allow_local_fallback and self.local_retriever is not None:
            local_res = self.local_retriever.query_clinical_guidelines(query_text=query_text, top_k=top_k)
            status = local_res.get("status")

            if status == "success":
                return {
                    "kb_references": local_res.get("kb_references", []),
                    "differential_diagnosis": local_res.get("differential_diagnosis", []),
                    "top_match_condition": local_res.get("top_match_condition", "Guideline Retrieved"),
                    "similarity_score": local_res.get("top_score"),
                    "vector_engine": "Local Semantic RAG (all-MiniLM-L6-v2 / 384-d in-memory)",
                    "matches_count": local_res.get("matches_count", 0),
                    "retrieval_mode": "local_semantic",
                    "retrieved_chunks": local_res.get("retrieved_chunks", [])
                }
            elif status == "abstained":
                return {
                    "kb_references": [],
                    "differential_diagnosis": [],
                    "top_match_condition": "No Relevant Clinical Guideline Found",
                    "similarity_score": None,
                    "vector_engine": "Unavailable / No Matching Evidence",
                    "matches_count": 0,
                    "retrieval_mode": "abstained",
                    "retrieved_chunks": [],
                    "notice": local_res.get("notice", "Local semantic retrieval abstained: query below relevance threshold.")
                }
            elif status == "error":
                return {
                    "kb_references": [],
                    "differential_diagnosis": [],
                    "top_match_condition": "Offline RAG Dependency Error",
                    "similarity_score": None,
                    "vector_engine": "Unavailable / Dependency Error",
                    "matches_count": 0,
                    "retrieval_mode": "dependency_error",
                    "retrieved_chunks": [],
                    "error": local_res.get("error", "Unknown error in local retrieval")
                }

        # --------------------------------------------------------------------
        # Priority 3: Truthful Abstention / Unconfigured
        # --------------------------------------------------------------------
        return {
            "kb_references": [],
            "differential_diagnosis": [],
            "top_match_condition": "Offline / Guideline Index Unconfigured",
            "similarity_score": None,
            "vector_engine": "Unavailable / No Matching Evidence",
            "matches_count": 0,
            "retrieval_mode": "unconfigured" if not self.api_key else "unavailable",
            "retrieved_chunks": [],
            "notice": "Cloud and local clinical guideline retrieval are offline. No synthetic fallback scores injected."
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
