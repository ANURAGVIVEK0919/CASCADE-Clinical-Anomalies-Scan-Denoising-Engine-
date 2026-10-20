"""
CASCADE Structural Document Chunker

Splits parsed clinical guideline documents into overlapping chunks.
Generates deterministic, stable chunk IDs to prevent duplicate accumulation during re-ingestion.
"""

import hashlib
import re
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
        """
        Splits a parsed document into structured DocumentChunk objects.
        Uses section-aware Markdown chunking when Markdown headers are present,
        falling back to sliding window chunking for unstructured plain text.
        """
        if not doc_dict.get("success"):
            return []

        text = doc_dict.get("text", "")
        # If document has Markdown header structure, use section-aware chunking
        if any(line.strip().startswith("# ") or line.strip().startswith("## ") for line in text.splitlines()[:20]):
            return self.chunk_markdown_sections(doc_dict)

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
                        "raw_text": chunk_text,
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

    def chunk_markdown_sections(self, doc_dict: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Splits a structured Markdown clinical document into semantic, section-aware chunks.
        Preserves document title and section/subsection header hierarchy in each chunk,
        and extracts discrete clinical recommendation tiers without mid-sentence truncation.
        """
        if not doc_dict.get("success"):
            return []

        text = doc_dict["text"]
        source_file = doc_dict.get("filename", "unknown.txt")
        doc_title = doc_dict.get("title", source_file)
        doc_hash = doc_dict.get("content_hash", "")
        filepath = doc_dict.get("filepath", "")

        lines = text.splitlines()
        chunks: List[DocumentChunk] = []
        chunk_idx = 0

        main_title = doc_title
        current_h2 = ""
        current_h3 = ""
        current_lines = []

        def emit(body_lines: List[str], h2_ctx: str, h3_ctx: str, item_title: str = None):
            nonlocal chunk_idx
            body = "\n".join(body_lines).strip()
            if not body:
                return

            # Skip pure bibliographic preamble before first section or explicit Metadata sections
            if not h2_ctx and not item_title:
                return
            if h2_ctx.lower() in ("metadata", "license / terms", "source reference"):
                return

            header_lines = [f"# {main_title}"]
            if h2_ctx:
                header_lines.append(f"## {h2_ctx}")
            if h3_ctx:
                if item_title and item_title != h3_ctx:
                    header_lines.append(f"### {h3_ctx} — {item_title}")
                else:
                    header_lines.append(f"### {h3_ctx}")
            elif item_title:
                header_lines.append(f"### {item_title}")

            header_prefix = "\n".join(header_lines)
            full_content = f"{header_prefix}\n\n{body}"

            display_title = item_title or h3_ctx or h2_ctx or main_title
            chunk_id = self.generate_stable_chunk_id(source_file, chunk_idx, full_content)

            chunk_obj = DocumentChunk(
                chunk_id=chunk_id,
                content=full_content,
                source_file=source_file,
                title=f"{main_title}: {display_title}" if display_title != main_title else main_title,
                chunk_index=chunk_idx,
                start_char=0,
                end_char=len(body),
                content_hash=doc_hash,
                metadata={
                    "filename": source_file,
                    "title": display_title,
                    "document_title": main_title,
                    "section": h2_ctx,
                    "subsection": h3_ctx,
                    "item_title": item_title or "",
                    "raw_text": body,
                    "char_length": len(full_content),
                    "source_path": filepath
                }
            )
            chunks.append(chunk_obj)
            chunk_idx += 1

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith("# "):
                if current_lines:
                    emit(current_lines, current_h2, current_h3)
                    current_lines = []
                main_title = stripped[2:].strip()
                current_h2 = ""
                current_h3 = ""
                i += 1
            elif stripped.startswith("## "):
                if current_lines:
                    emit(current_lines, current_h2, current_h3)
                    current_lines = []
                current_h2 = stripped[3:].strip()
                current_h3 = ""
                i += 1
            elif stripped.startswith("### "):
                if current_lines:
                    emit(current_lines, current_h2, current_h3)
                    current_lines = []
                current_h3 = stripped[4:].strip()
                i += 1
            # For clinical guidelines with structured size categories under Single Solid Nodules
            elif current_h3 == "Single Solid Nodules" and stripped.startswith("* **Nodules "):
                if current_lines:
                    emit(current_lines, current_h2, current_h3)
                    current_lines = []

                bullet_lines = [line]
                m = re.match(r"\*\s+\*\*([^*]+)\*\*:", stripped)
                bullet_title = m.group(1).strip() if m else None

                i += 1
                while i < n and (lines[i].startswith("  ") or lines[i].startswith("\t") or not lines[i].strip()):
                    if lines[i].strip() or (i + 1 < n and (lines[i+1].startswith("  ") or lines[i+1].startswith("\t"))):
                        bullet_lines.append(lines[i])
                    i += 1
                emit(bullet_lines, current_h2, current_h3, item_title=bullet_title)
            else:
                current_lines.append(line)
                i += 1

        if current_lines:
            emit(current_lines, current_h2, current_h3)

        return chunks
