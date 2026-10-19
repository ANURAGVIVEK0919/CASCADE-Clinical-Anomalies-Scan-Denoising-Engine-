"""
CASCADE Pinecone Read-Only Connection & Index Verifier

Non-destructive diagnostic CLI:
- Verifies Pinecone API connectivity and index status without writing or altering data.
- Inspects remote index dimension, metric, host, and namespaced vector counts.
- Verifies embedding model compatibility (384-dimensional sentence-transformers).
- Strictly read-only: NEVER creates indexes, NEVER upserts, and NEVER deletes vectors.
- Sanitizes credentials: NEVER logs or prints PINECONE_API_KEY.
"""

import os
import sys
import argparse
from typing import Dict, Any

from rag.embeddings import DEFAULT_VECTOR_DIM, DEFAULT_NAMESPACE, SemanticEmbeddingEngine


def verify_pinecone_environment(
    index_name: str = "cascade-clinical-rag",
    namespace: str = DEFAULT_NAMESPACE,
    test_query: str = None
) -> Dict[str, Any]:
    """
    Connects to Pinecone in strictly read-only mode and inspects index metadata and stats.
    """
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    if not api_key:
        return {
            "status": "UNCONFIGURED",
            "message": "PINECONE_API_KEY environment variable is not set.",
            "instructions": "Set the environment variable securely: $env:PINECONE_API_KEY='your-key-here'",
            "is_connected": False
        }

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)

        # 1. List indexes (read-only)
        indexes_info = pc.list_indexes()
        index_names = [idx.name for idx in indexes_info]

        if index_name not in index_names:
            return {
                "status": "INDEX_NOT_FOUND",
                "message": f"Index '{index_name}' does not exist on this Pinecone account.",
                "available_indexes": index_names,
                "is_connected": True,
                "index_ready": False
            }

        # 2. Inspect target index configuration
        target_meta = next((idx for idx in indexes_info if idx.name == index_name), None)
        dim = getattr(target_meta, "dimension", None)
        metric = getattr(target_meta, "metric", "cosine")
        host = getattr(target_meta, "host", "unknown")

        dimension_match = (dim == DEFAULT_VECTOR_DIM)

        # 3. Connect to index and describe stats (read-only)
        index = pc.Index(index_name)
        stats = index.describe_index_stats()

        namespaces_dict = getattr(stats, "namespaces", {}) or stats.get("namespaces", {})
        target_ns_stats = namespaces_dict.get(namespace, {})
        vector_count = getattr(target_ns_stats, "vector_count", None)
        if vector_count is None and isinstance(target_ns_stats, dict):
            vector_count = target_ns_stats.get("vector_count", 0)

        total_vectors = getattr(stats, "total_vector_count", None)
        if total_vectors is None and isinstance(stats, dict):
            total_vectors = stats.get("total_vector_count", 0)

        result = {
            "status": "READY" if dimension_match else "DIMENSION_MISMATCH",
            "is_connected": True,
            "index_ready": True,
            "index_name": index_name,
            "target_namespace": namespace,
            "metric": metric,
            "dimension": dim,
            "expected_dimension": DEFAULT_VECTOR_DIM,
            "dimension_compatible": dimension_match,
            "total_vectors_in_index": total_vectors,
            "vectors_in_target_namespace": vector_count or 0,
            "host": host
        }

        # 4. Optional read-only sample query
        if test_query and result["status"] == "READY" and (vector_count or 0) > 0:
            embedder = SemanticEmbeddingEngine()
            q_vec = embedder.embed_text(test_query)
            query_res = index.query(
                vector=q_vec,
                top_k=2,
                include_metadata=True,
                namespace=namespace
            )
            matches = []
            for m in query_res.get("matches", []):
                meta = m.get("metadata", {})
                matches.append({
                    "id": m.get("id"),
                    "score": round(float(m.get("score", 0.0)), 4),
                    "title": meta.get("title", "Clinical Reference"),
                    "text_preview": (meta.get("text") or meta.get("content", ""))[:120] + "..."
                })
            result["sample_query_results"] = matches

        return result

    except Exception as e:
        return {
            "status": "CONNECTION_ERROR",
            "message": str(e),
            "is_connected": False
        }


def main():
    parser = argparse.ArgumentParser(description="Read-Only Pinecone Index & Namespace Inspector")
    parser.add_argument("--index", default=os.environ.get("PINECONE_INDEX_NAME", "cascade-clinical-rag"), help="Pinecone index name")
    parser.add_argument("--namespace", default=os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE), help="Target namespace")
    parser.add_argument("--query", default=None, help="Optional read-only test query text")
    args = parser.parse_args()

    print("=" * 65)
    print("  CASCADE PINECONE READ-ONLY STATUS & CONNECTION VERIFICATION")
    print("=" * 65)

    res = verify_pinecone_environment(index_name=args.index, namespace=args.namespace, test_query=args.query)

    print(f"Connection Status:       {res['status']}")
    if res.get("message"):
        print(f"Notice:                  {res['message']}")
    if res.get("instructions"):
        print(f"Instructions:            {res['instructions']}")

    if res.get("is_connected") and res.get("index_ready"):
        print(f"Index Name:              {res['index_name']}")
        print(f"Host:                    {res.get('host')}")
        print(f"Target Namespace:        {res['target_namespace']}")
        print(f"Metric:                  {res['metric']}")
        print(f"Index Dimension:         {res['dimension']} (Expected: {res['expected_dimension']})")
        print(f"Dimension Compatible:    {res['dimension_compatible']}")
        print(f"Total Vectors in Index:  {res['total_vectors_in_index']}")
        print(f"Vectors in Target NS:    {res['vectors_in_target_namespace']}")

        if "sample_query_results" in res:
            print("\nSample Read-Only Query Results:")
            for idx, match in enumerate(res["sample_query_results"], 1):
                print(f"  {idx}. [{match['score']:.4f}] {match['title']} (ID: {match['id']})")
                print(f"     Preview: {match['text_preview']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
