"""
CASCADE Clinical Document Ingestion CLI & Pipeline

Explicit command-line interface for ingesting clinical guideline documents into Pinecone:
- Configurable source directory (default: knowledge_base/clinical_sources/)
- Multi-format parsing (.md, .txt, .pdf, .html)
- Deterministic chunking with stable IDs
- Batched semantic embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Explicit batch upsert to designated Pinecone namespace
- Dry-run mode for offline verification
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Any

from .parsers import DocumentParser
from .chunker import DocumentChunker, DocumentChunk
from .embeddings import SemanticEmbeddingEngine, DEFAULT_NAMESPACE, DEFAULT_VECTOR_DIM


def ingest_directory(
    source_dir: str = "knowledge_base/clinical_sources",
    namespace: str = DEFAULT_NAMESPACE,
    batch_size: int = 32,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Ingests all supported documents from source_dir into Pinecone namespace.
    """
    print("=" * 65)
    print("  CASCADE CLINICAL GUIDELINES RAG INGESTION PIPELINE")
    print("=" * 65)
    print(f"Source Directory:  {source_dir}")
    print(f"Target Namespace:  {namespace}")
    print(f"Dry Run Mode:      {dry_run}")

    if not os.path.exists(source_dir):
        os.makedirs(source_dir, exist_ok=True)
        print(f"[*] Created empty clinical source directory: {source_dir}")

    # Discover files
    supported_exts = {".txt", ".md", ".pdf", ".html", ".htm"}
    files = []
    for root, _, filenames in os.walk(source_dir):
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            if ext in supported_exts:
                files.append(os.path.join(root, f))

    print(f"[*] Discovered {len(files)} candidate source documents.")

    if not files:
        return {
            "status": "no_documents",
            "message": f"No supported documents (.md, .txt, .pdf, .html) found in {source_dir}.",
            "documents_parsed": 0,
            "chunks_extracted": 0,
            "chunks_upserted": 0
        }

    # 1. Parse documents
    parser = DocumentParser()
    parsed_docs = []
    parse_errors = []

    for fpath in files:
        res = parser.parse_file(fpath)
        if res.get("success"):
            parsed_docs.append(res)
        else:
            parse_errors.append((fpath, res.get("error", "Unknown error")))

    print(f"[*] Successfully parsed {len(parsed_docs)} documents ({len(parse_errors)} errors).")
    for err_path, err_msg in parse_errors:
        print(f"    [!] Error parsing {os.path.basename(err_path)}: {err_msg}")

    # 2. Chunk documents
    chunker = DocumentChunker(chunk_size=600, chunk_overlap=120)
    all_chunks: List[DocumentChunk] = []

    for pdoc in parsed_docs:
        doc_chunks = chunker.chunk_document(pdoc)
        all_chunks.extend(doc_chunks)

    print(f"[*] Extracted {len(all_chunks)} semantic chunks across {len(parsed_docs)} documents.")

    if not all_chunks:
        return {
            "status": "empty_chunks",
            "message": "Documents produced zero valid text chunks.",
            "documents_parsed": len(parsed_docs),
            "chunks_extracted": 0,
            "chunks_upserted": 0
        }

    # 3. Generate Embeddings
    print("[*] Initializing semantic embedding engine (sentence-transformers/all-MiniLM-L6-v2)...")
    embedder = SemanticEmbeddingEngine()
    chunk_texts = [c.content for c in all_chunks]

    start_embed = time.time()
    embeddings = embedder.embed_batch(chunk_texts, batch_size=batch_size)
    embed_time = time.time() - start_embed
    print(f"[*] Generated {len(embeddings)} vectors ({DEFAULT_VECTOR_DIM}-d) in {embed_time:.2f}s.")

    if dry_run:
        print("[*] Dry-run enabled. Skipping remote Pinecone upsert.")
        return {
            "status": "dry_run_success",
            "documents_parsed": len(parsed_docs),
            "chunks_extracted": len(all_chunks),
            "chunks_embedded": len(embeddings),
            "chunks_upserted": 0,
            "embedding_model": embedder.model_name,
            "vector_dimension": DEFAULT_VECTOR_DIM,
            "namespace": namespace
        }

    # 4. Upsert to Pinecone
    api_key = os.environ.get("PINECONE_API_KEY", "").strip()
    index_name = os.environ.get("PINECONE_INDEX_NAME", "cascade-clinical-rag").strip()

    if not api_key:
        print("[!] Notice: PINECONE_API_KEY is not configured in environment.")
        print("[!] Ingestion completed locally; remote upsert bypassed.")
        return {
            "status": "completed_offline",
            "reason": "PINECONE_API_KEY not configured",
            "documents_parsed": len(parsed_docs),
            "chunks_extracted": len(all_chunks),
            "chunks_embedded": len(embeddings),
            "chunks_upserted": 0
        }

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)

        # Batch upsert
        upsert_records = []
        for chunk, vec in zip(all_chunks, embeddings):
            upsert_records.append({
                "id": chunk.chunk_id,
                "values": vec,
                "metadata": {
                    "filename": chunk.source_file,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_hash": chunk.content_hash
                }
            })

        print(f"[*] Upserting {len(upsert_records)} records to Pinecone index '{index_name}' (namespace: '{namespace}')...")
        for i in range(0, len(upsert_records), batch_size):
            batch = upsert_records[i : i + batch_size]
            index.upsert(vectors=batch, namespace=namespace)

        print(f"[SUCCESS] Ingested and upserted {len(upsert_records)} vectors to Pinecone namespace '{namespace}'.")
        return {
            "status": "success",
            "documents_parsed": len(parsed_docs),
            "chunks_extracted": len(all_chunks),
            "chunks_upserted": len(upsert_records),
            "index_name": index_name,
            "namespace": namespace
        }

    except Exception as e:
        print(f"[!] Error upserting to Pinecone: {e}")
        return {
            "status": "error",
            "error": str(e),
            "documents_parsed": len(parsed_docs),
            "chunks_extracted": len(all_chunks),
            "chunks_upserted": 0
        }


def main():
    parser = argparse.ArgumentParser(description="CASCADE Clinical Guideline Ingestion")
    parser.add_argument("--source-dir", default="knowledge_base/clinical_sources", help="Directory containing documents")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="Target Pinecone namespace")
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding and upsert batch size")
    parser.add_argument("--dry-run", action="store_true", help="Parse and embed without upserting to Pinecone")
    args = parser.parse_args()

    res = ingest_directory(
        source_dir=args.source_dir,
        namespace=args.namespace,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
    print(f"\nFinal Status: {res.get('status')}")


if __name__ == "__main__":
    main()
