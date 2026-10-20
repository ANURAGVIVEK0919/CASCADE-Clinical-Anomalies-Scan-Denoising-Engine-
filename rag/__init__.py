"""
CASCADE RAG Package

Enterprise Clinical RAG Module:
- Semantic Embeddings (sentence-transformers/all-MiniLM-L6-v2, 384-d, L2 normalized)
- Document Ingestion (PDF, TXT, MD, HTML with stable SHA-256 IDs)
- Genuine Namespaced Pinecone Retrieval (no synthetic score bonuses)
- Grounded Retrieval Evaluation
"""

from .embeddings import SemanticEmbeddingEngine
from .chunker import DocumentChunker, DocumentChunk
from .parsers import DocumentParser
from .retriever import ClinicalRAGRetriever
from .local_retriever import LocalClinicalRAGRetriever

__all__ = [
    "SemanticEmbeddingEngine",
    "DocumentChunker",
    "DocumentChunk",
    "DocumentParser",
    "ClinicalRAGRetriever",
    "LocalClinicalRAGRetriever"
]
