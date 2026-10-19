"""
CASCADE Document Parsers for Clinical Guideline Ingestion

Extracts clean textual content and structural metadata from:
- Plain Text (.txt)
- Markdown (.md)
- Adobe PDF (.pdf via pypdf)
- HTML (.html, .htm via BeautifulSoup)

Never executes document content as code.
"""

import os
import hashlib
from typing import Dict, Any, Optional


class DocumentParser:
    """Multi-format clinical document reader."""

    @staticmethod
    def compute_sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def parse_file(cls, filepath: str) -> Dict[str, Any]:
        """Reads file, extracts text, and generates standardized document metadata."""
        if not os.path.exists(filepath):
            return {"error": f"File not found: {filepath}", "success": False}

        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        size_bytes = os.path.getsize(filepath)

        if size_bytes == 0:
            return {"error": f"File is empty (0 bytes): {filename}", "success": False}

        try:
            if ext in [".txt", ".md"]:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read()
                title = None
                for line in text.splitlines():
                    s = line.strip()
                    if s.startswith("# "):
                        title = s[2:].strip()
                        break
                return cls._build_doc(filepath, text, ext, size_bytes, title=title)

            elif ext == ".pdf":
                try:
                    import pypdf
                    reader = pypdf.PdfReader(filepath)
                    pages_text = []
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text() or ""
                        pages_text.append(f"--- Page {i+1} ---\n" + page_text)
                    full_text = "\n\n".join(pages_text)
                    return cls._build_doc(filepath, full_text, ext, size_bytes, num_pages=len(reader.pages))
                except Exception as e:
                    return {"error": f"PDF parse error ({filename}): {e}", "success": False}

            elif ext in [".html", ".htm"]:
                try:
                    from bs4 import BeautifulSoup
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        soup = BeautifulSoup(f.read(), "html.parser")
                    # Extract title
                    title_tag = soup.find("title")
                    title = title_tag.text.strip() if title_tag else filename
                    # Remove scripts and styles
                    for s in soup(["script", "style"]):
                        s.extract()
                    text = soup.get_text(separator="\n").strip()
                    return cls._build_doc(filepath, text, ext, size_bytes, title=title)
                except Exception as e:
                    return {"error": f"HTML parse error ({filename}): {e}", "success": False}

            else:
                return {"error": f"Unsupported file format '{ext}' for file {filename}", "success": False}

        except Exception as e:
            return {"error": f"Failed reading file {filename}: {e}", "success": False}

    @classmethod
    def _build_doc(cls, filepath: str, text: str, ext: str, size_bytes: int, **extra) -> Dict[str, Any]:
        text = text.strip()
        if not text:
            return {"error": f"No textual content could be extracted from {filepath}", "success": False}

        filename = os.path.basename(filepath)
        title = extra.get("title") or os.path.splitext(filename)[0].replace("_", " ").title()

        return {
            "success": True,
            "filepath": filepath,
            "filename": filename,
            "title": title,
            "extension": ext,
            "size_bytes": size_bytes,
            "char_count": len(text),
            "content_hash": cls.compute_sha256(text),
            "text": text,
            "extra_metadata": extra
        }
