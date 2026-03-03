from __future__ import annotations

import hashlib
import re
from typing import List

from src.models import BoundingBox, ExtractedDocument, LDU, ChunkType


def _approx_token_count(text: str) -> int:
    # Very rough approximation: 1 token ~= 4 characters
    return max(0, int(len(text) / 4))


def _hash_content(doc_id: str, page_numbers: List[int], text: str) -> str:
    h = hashlib.sha256()
    h.update(doc_id.encode("utf-8"))
    h.update(",".join(str(p) for p in sorted(page_numbers)).encode("utf-8"))
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class ChunkValidator:
    """Verifies that core chunking rules are not violated."""

    def validate(self, chunks: List[LDU]) -> None:
        # For this initial implementation, we only perform basic sanity checks.
        for chunk in chunks:
            assert chunk.content, "Chunk content must not be empty"
            assert chunk.token_count >= 0


class ChunkingEngine:
    """
    Semantic Chunking Engine.

    Rules enforced:
    - Table cells are never split from their header row.
    - Figure captions are stored with their figure chunk.
    - Numbered lists are kept as a single LDU unless they exceed max_tokens.
    - Section headers are stored as parent metadata (placeholder – requires PageIndex).
    - Cross-references can be added later as relationships.
    """

    def __init__(self, max_tokens_per_chunk: int = 512) -> None:
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.validator = ChunkValidator()

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        chunks: List[LDU] = []

        # Paragraph-like chunks from text blocks, with list detection & cross-refs
        list_pattern = re.compile(r"^(\d+[\.\)]|[a-zA-Z][\.\)])\s+")
        xref_pattern = re.compile(r"\b(Table|Figure)\s+\d+(\.\d+)?", re.IGNORECASE)

        for tb in doc.text_blocks:
            raw = tb.text or ""
            text = raw.strip()
            if not text:
                continue

            lines = [ln for ln in raw.splitlines() if ln.strip()]
            is_numbered_list = lines and all(list_pattern.match(ln.strip()) for ln in lines)

            if is_numbered_list:
                # Keep whole list as a single LDU unless extremely long
                tokens = _approx_token_count(text)
                if tokens > self.max_tokens_per_chunk:
                    # Fallback to paragraph splitting if list is too long
                    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    source_type = ChunkType.list
                else:
                    parts = [text]
                    source_type = ChunkType.list
            else:
                tokens = _approx_token_count(text)
                if tokens > self.max_tokens_per_chunk:
                    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    source_type = ChunkType.paragraph
                else:
                    parts = [text]
                    source_type = ChunkType.paragraph

            for part in parts:
                if not part:
                    continue
                ptokens = _approx_token_count(part)
                page_refs = [tb.page_number]
                content_hash = _hash_content(doc.doc_id, page_refs, part)
                refs = list({m.group(0) for m in xref_pattern.finditer(part)})
                chunks.append(
                    LDU(
                        id=content_hash[:16],
                        doc_id=doc.doc_id,
                        chunk_type=source_type,
                        content=part,
                        page_refs=page_refs,
                        bounding_box=tb.bbox,
                        parent_section=None,
                        token_count=ptokens,
                        content_hash=content_hash,
                        references=refs,
                    )
                )

        # Table chunks: keep headers + all rows together as a single LDU
        for tbl in doc.tables:
            header_line = " | ".join(tbl.headers) if tbl.headers else ""
            row_lines = []
            for row in tbl.rows:
                row_lines.append(" | ".join(cell.text for cell in row))
            content = "\n".join(
                [header_line] + row_lines if header_line else row_lines
            ).strip()
            if not content:
                continue
            tokens = _approx_token_count(content)
            page_refs = [tbl.page_number]
            bbox = tbl.bbox or BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
            content_hash = _hash_content(doc.doc_id, page_refs, content)
            chunks.append(
                LDU(
                    id=content_hash[:16],
                    doc_id=doc.doc_id,
                    chunk_type=ChunkType.table,
                    content=content,
                    page_refs=page_refs,
                    bounding_box=bbox,
                    parent_section=None,
                    token_count=tokens,
                    content_hash=content_hash,
                    references=[],
                )
            )

        # Figures with captions as metadata of figure chunks
        for fig in doc.figures:
            caption = fig.caption or ""
            if not caption.strip():
                continue
            tokens = _approx_token_count(caption)
            page_refs = [fig.page_number]
            bbox = fig.bbox or BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
            content_hash = _hash_content(doc.doc_id, page_refs, caption)
            chunks.append(
                LDU(
                    id=content_hash[:16],
                    doc_id=doc.doc_id,
                    chunk_type=ChunkType.figure,
                    content=caption,
                    page_refs=page_refs,
                    bounding_box=bbox,
                    parent_section=None,
                    token_count=tokens,
                    content_hash=content_hash,
                    references=[],
                )
            )

        self.validator.validate(chunks)
        return chunks

