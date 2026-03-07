from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from src.models import BoundingBox, ExtractedDocument, LDU, ChunkType
from src.utils.content_hash import content_hash


class ChunkValidationError(Exception):
    """Raised when ChunkValidator finds a violation of the five chunking rules."""

    def __init__(self, message: str, chunk_id: str = "") -> None:
        self.chunk_id = chunk_id
        super().__init__(message)


def _approx_token_count(text: str) -> int:
    # Very rough approximation: 1 token ~= 4 characters
    return max(0, int(len(text) / 4))


class ChunkValidator:
    """
    Verifies that the five core chunking rules are not violated:
    1. Table-header binding: table LDUs contain header + rows (never split).
    2. Figure captions: figure LDUs have non-empty caption content.
    3. Numbered lists: list LDUs do not exceed hard_max_tokens (kept intact when possible).
    4. Section header propagation: parent_section may be set (optional at chunk time).
    5. Cross-references: references that match known tables/figures have resolved relationships.
    """

    def __init__(self, hard_max_tokens_per_chunk: int = 1024) -> None:
        self.hard_max_tokens_per_chunk = hard_max_tokens_per_chunk

    def validate(self, chunks: List[LDU]) -> None:
        table_ids = [c.id for c in chunks if c.chunk_type == ChunkType.table]
        figure_ids = [c.id for c in chunks if c.chunk_type == ChunkType.figure]
        ref_to_id: Dict[str, str] = {}
        for i, ldu_id in enumerate(table_ids, start=1):
            ref_to_id[f"Table {i}"] = ldu_id
            ref_to_id[f"table {i}"] = ldu_id
        for i, ldu_id in enumerate(figure_ids, start=1):
            ref_to_id[f"Figure {i}"] = ldu_id
            ref_to_id[f"figure {i}"] = ldu_id

        for chunk in chunks:
            if not chunk.content:
                raise ChunkValidationError("Chunk content must not be empty", chunk.id)
            if chunk.token_count < 0:
                raise ChunkValidationError("Chunk token_count must be non-negative", chunk.id)
            if chunk.chunk_type == ChunkType.table:
                if "\n" not in chunk.content and "|" not in chunk.content:
                    raise ChunkValidationError(
                        "Table LDU must contain header and/or rows (table-header binding).",
                        chunk.id,
                    )
            if chunk.chunk_type == ChunkType.figure:
                if not chunk.content.strip():
                    raise ChunkValidationError(
                        "Figure LDU must have non-empty caption content.",
                        chunk.id,
                    )
            if chunk.chunk_type == ChunkType.list:
                if chunk.token_count > self.hard_max_tokens_per_chunk:
                    raise ChunkValidationError(
                        "List LDU must not exceed hard_max_tokens (numbered lists kept intact).",
                        chunk.id,
                    )
            for ref in chunk.references:
                ref_key = ref.strip()
                if ref_key in ref_to_id:
                    target_id = ref_to_id[ref_key]
                    resolved = [
                        r for r in (chunk.relationships or [])
                        if r.get("target_ldu_id") == target_id
                    ]
                    if not resolved:
                        raise ChunkValidationError(
                            f"Cross-reference '{ref}' should resolve to LDU id {target_id}.",
                            chunk.id,
                        )


@dataclass
class ChunkingConfig:
    max_tokens_per_chunk: int
    hard_max_tokens_per_chunk: int
    keep_numbered_lists_together: bool


def _resolve_cross_references(chunks: List[LDU]) -> None:
    """Resolve reference strings (e.g. 'Table 3', 'Figure 2') to target LDU ids."""
    table_ids = [c.id for c in chunks if c.chunk_type == ChunkType.table]
    figure_ids = [c.id for c in chunks if c.chunk_type == ChunkType.figure]
    ref_to_id: Dict[str, str] = {}
    for i, ldu_id in enumerate(table_ids, start=1):
        ref_to_id["Table %d" % i] = ldu_id
        ref_to_id["table %d" % i] = ldu_id
    for i, ldu_id in enumerate(figure_ids, start=1):
        ref_to_id["Figure %d" % i] = ldu_id
        ref_to_id["figure %d" % i] = ldu_id

    for chunk in chunks:
        if not chunk.references:
            continue
        resolved: List[Dict] = []
        for ref in chunk.references:
            key = ref.strip()
            if key in ref_to_id:
                resolved.append({
                    "target_ldu_id": ref_to_id[key],
                    "ref_text": ref,
                })
            else:
                # Try normalized form (e.g. "Figure 2.1" -> "Figure 2")
                m = re.match(r"(?i)(Table|Figure)\s+(\d+)", key)
                if m:
                    prefix, num = m.group(1), m.group(2)
                    canonical = f"{prefix.capitalize()} {num}"
                    if canonical in ref_to_id:
                        resolved.append({
                            "target_ldu_id": ref_to_id[canonical],
                            "ref_text": ref,
                        })
        if resolved:
            chunk.relationships = resolved


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
        self.validator = ChunkValidator(
            hard_max_tokens_per_chunk=self.hard_max_tokens_per_chunk,
        )

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
                content_hash_val = content_hash(doc.doc_id, page_refs, part)
                if content_hash_val in seen_hashes:
                    continue
                seen_hashes.add(content_hash_val)
                refs = list({m.group(0) for m in xref_pattern.finditer(part)})
                chunks.append(
                    LDU(
                        id=content_hash_val,
                        doc_id=doc.doc_id,
                        chunk_type=source_type,
                        content=part,
                        page_refs=page_refs,
                        bounding_box=tb.bbox,
                        parent_section=None,
                        token_count=ptokens,
                        content_hash=content_hash_val,
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
            content_hash_val = content_hash(doc.doc_id, page_refs, content)
            if content_hash_val in seen_hashes:
                continue
            seen_hashes.add(content_hash_val)
            chunks.append(
                LDU(
                    id=content_hash_val,
                    doc_id=doc.doc_id,
                    chunk_type=ChunkType.table,
                    content=content,
                    page_refs=page_refs,
                    bounding_box=bbox,
                    parent_section=None,
                    token_count=tokens,
                    content_hash=content_hash_val,
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
            content_hash_val = content_hash(doc.doc_id, page_refs, caption)
            if content_hash_val in seen_hashes:
                continue
            seen_hashes.add(content_hash_val)
            chunks.append(
                LDU(
                    id=content_hash_val,
                    doc_id=doc.doc_id,
                    chunk_type=ChunkType.figure,
                    content=caption,
                    page_refs=page_refs,
                    bounding_box=bbox,
                    parent_section=None,
                    token_count=tokens,
                    content_hash=content_hash_val,
                    references=[],
                )
            )

        # Resolve cross-references to LDU ids (Table N, Figure N by document order).
        _resolve_cross_references(chunks)

        self.validator.validate(chunks)
        return chunks

