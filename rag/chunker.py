"""
CASCADE Structural Document Chunker

Splits parsed clinical guideline documents into overlapping chunks.
Generates deterministic, stable chunk IDs to prevent duplicate accumulation during re-ingestion.
"""

import hashlib
from typing import List, Dict, Any


class DocumentChunk:
    """Represents an individual chunk of clinical knowledge ready for embedding and retrieval."""

    def __init__(
        self,
        chunk_id: str,
        content: str,
        source_file: str,
        title: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        content_hash: str,
        metadata: Dict[str, Any] = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.source_file = source_file
        self.title = title
        self.chunk_index = chunk_index
        self.start_char = start_char
        self.end_char = end_char
        self.content_hash = content_hash
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_file": self.source_file,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "content_hash": self.content_hash,
            "metadata": self.metadata
        }


class DocumentChunker:
    """Splits documents with configurable chunk sizes and overlaps."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def generate_stable_chunk_id(source_file: str, chunk_index: int, content_snippet: str) -> str:
        """
        Creates a deterministic 16-character SHA-256 chunk identifier.
        Rerunning ingestion on the same file produces identical IDs.
        """
        unique_key = f"{source_file}::chunk_{chunk_index:04d}::{content_snippet[:64]}"
        return hashlib.sha256(unique_key.encode("utf-8")).hexdigest()[:16]

    def chunk_document(self, doc_dict: Dict[str, Any]) -> List[DocumentChunk]:
        """Splits a parsed document into structured DocumentChunk objects."""
        if not doc_dict.get("success"):
            return []

        text = doc_dict["text"]
        source_file = doc_dict.get("filename", "unknown.txt")
        title = doc_dict.get("title", source_file)
        doc_hash = doc_dict.get("content_hash", "")

        chunks = []
        start = 0
        text_len = len(text)
        chunk_idx = 0

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Try not to break words mid-sentence if not at end
            if end < text_len:
                last_space = text.rfind(" ", start, end)
                if last_space > start + (self.chunk_size // 2):
                    end = last_space

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunk_id = self.generate_stable_chunk_id(source_file, chunk_idx, chunk_text)
                chunk_obj = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    source_file=source_file,
                    title=title,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                    content_hash=doc_hash,
                    metadata={
                        "filename": source_file,
                        "title": title,
                        "char_length": len(chunk_text),
                        "source_path": doc_dict.get("filepath", "")
                    }
                )
                chunks.append(chunk_obj)
                chunk_idx += 1

            if end >= text_len:
                break
            start = end - self.chunk_overlap
            if start < 0:
                start = 0

        return chunks
