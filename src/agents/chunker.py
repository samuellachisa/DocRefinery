from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml

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


@dataclass
class ChunkingConfig:
    max_tokens_per_chunk: int
    hard_max_tokens_per_chunk: int
    keep_numbered_lists_together: bool


def load_chunking_config(config_path: Path) -> ChunkingConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = data.get("chunking", {}) or {}
    return ChunkingConfig(
        max_tokens_per_chunk=int(cfg.get("max_tokens_per_chunk", 512)),
        hard_max_tokens_per_chunk=int(cfg.get("hard_max_tokens_per_chunk", 1024)),
        keep_numbered_lists_together=bool(cfg.get("keep_numbered_lists_together", True)),
    )


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

    def __init__(self, rules_path: Path | None = None) -> None:
        if rules_path is None:
            rules_path = Path("rubric/extraction_rules.yaml")
        cfg = load_chunking_config(rules_path)
        self.max_tokens_per_chunk = cfg.max_tokens_per_chunk
        self.hard_max_tokens_per_chunk = cfg.hard_max_tokens_per_chunk
        self.keep_numbered_lists_together = cfg.keep_numbered_lists_together
        self.validator = ChunkValidator()

    def _split_hard_max(self, text: str) -> List[str]:
        """
        Ensure that no resulting piece exceeds the hard_max_tokens_per_chunk.
        Uses a simple recursive halving strategy based on approximate token count.
        """
        tokens = _approx_token_count(text)
        if tokens <= self.hard_max_tokens_per_chunk:
            return [text]

        # Split on paragraph boundaries first, then fall back to character-level halving.
        if "\n\n" in text:
            parts = [p for p in text.split("\n\n") if p.strip()]
            result: List[str] = []
            for part in parts:
                result.extend(self._split_hard_max(part))
            return result

        # Fallback: naive halving of the string until within budget.
        mid = max(1, len(text) // 2)
        left = text[:mid]
        right = text[mid:]
        result: List[str] = []
        if left.strip():
            result.extend(self._split_hard_max(left))
        if right.strip():
            result.extend(self._split_hard_max(right))
        return result

    def chunk(self, doc: ExtractedDocument) -> List[LDU]:
        chunks: List[LDU] = []
        seen_hashes: set[str] = set()

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
                tokens = _approx_token_count(text)
                if (
                    self.keep_numbered_lists_together
                    and tokens <= self.hard_max_tokens_per_chunk
                ):
                    # Prefer to keep the whole list together when within hard budget.
                    parts = [text]
                    source_type = ChunkType.list
                elif tokens > self.max_tokens_per_chunk:
                    # Fallback to paragraph splitting if list is too long for normal chunks.
                    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    parts = []
                    for rp in raw_parts:
                        parts.extend(self._split_hard_max(rp))
                    source_type = ChunkType.list
                else:
                    parts = [text]
                    source_type = ChunkType.list
            else:
                tokens = _approx_token_count(text)
                if tokens > self.max_tokens_per_chunk:
                    raw_parts = [p.strip() for p in text.split("\n\n") if p.strip()]
                    parts = []
                    for rp in raw_parts:
                        parts.extend(self._split_hard_max(rp))
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
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)
                refs = list({m.group(0) for m in xref_pattern.finditer(part)})
                chunks.append(
                    LDU(
                        id=content_hash,
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
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunks.append(
                LDU(
                    id=content_hash,
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
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunks.append(
                LDU(
                    id=content_hash,
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

