"""Stable content hash for provenance and deduplication."""

from __future__ import annotations

import hashlib
from typing import List


def content_hash(doc_id: str, page_numbers: List[int], text: str) -> str:
    """Produce a stable SHA-256 hash over (doc_id, pages, content)."""
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(",".join(str(p) for p in sorted(page_numbers)).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()
