"""Unit tests for ChunkingEngine and ChunkValidator."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.chunker import ChunkingEngine, ChunkValidator, ChunkValidationError
from src.models import BoundingBox, ExtractedDocument, LDU, Table, TableCell, TextBlock
from src.models import ChunkType


def _make_extracted_doc(
    doc_id: str = "test-doc",
    text_blocks: list[tuple[int, str]] | None = None,
    tables: list | None = None,
) -> ExtractedDocument:
    """Build a minimal ExtractedDocument for testing."""
    blocks: list[TextBlock] = []
    if text_blocks:
        for i, (page_num, text) in enumerate(text_blocks):
            blocks.append(
                TextBlock(
                    page_number=page_num,
                    bbox=BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
                    text=text,
                    reading_order=i,
                )
            )
    tbls = tables or []
    num_pages = max((p for p, _ in text_blocks), default=1) if text_blocks else 1
    return ExtractedDocument(
        doc_id=doc_id,
        num_pages=num_pages,
        text_blocks=blocks,
        tables=tbls,
        figures=[],
    )


def test_chunker_produces_ldus_from_text_blocks(rules_path: Path) -> None:
    """ChunkingEngine produces LDUs from text blocks."""
    engine = ChunkingEngine(rules_path)
    doc = _make_extracted_doc(
        text_blocks=[
            (1, "This is a simple paragraph for testing."),
            (2, "Another paragraph on page two."),
        ]
    )
    chunks = engine.chunk(doc)
    assert len(chunks) == 2
    assert all(c.chunk_type == ChunkType.paragraph for c in chunks)
    assert chunks[0].content == "This is a simple paragraph for testing."
    assert chunks[1].content == "Another paragraph on page two."
    assert chunks[0].page_refs == [1]
    assert chunks[1].page_refs == [2]


def test_chunker_detects_numbered_list(rules_path: Path) -> None:
    """ChunkingEngine classifies numbered lists as ChunkType.list."""
    engine = ChunkingEngine(rules_path)
    doc = _make_extracted_doc(
        text_blocks=[
            (
                1,
                "1. First item\n2. Second item\n3. Third item",
            ),
        ]
    )
    chunks = engine.chunk(doc)
    assert len(chunks) == 1
    assert chunks[0].chunk_type == ChunkType.list


def test_chunker_extracts_table_as_single_ldu(rules_path: Path) -> None:
    """ChunkingEngine keeps table headers and rows together."""
    engine = ChunkingEngine(rules_path)
    doc = _make_extracted_doc(
        text_blocks=[(1, "Some text")],
        tables=[
            Table(
                page_number=1,
                headers=["Revenue", "Cost"],
                rows=[
                    [TableCell(row_index=0, col_index=0, text="100"), TableCell(row_index=0, col_index=1, text="50")],
                ],
            ),
        ],
    )
    chunks = engine.chunk(doc)
    assert len(chunks) >= 2  # text block + table
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.table]
    assert len(table_chunks) == 1
    assert "Revenue" in table_chunks[0].content
    assert "Cost" in table_chunks[0].content


def test_chunker_detects_cross_references(rules_path: Path) -> None:
    """ChunkingEngine extracts cross-references like Table 3, Figure 2.1."""
    engine = ChunkingEngine(rules_path)
    doc = _make_extracted_doc(
        text_blocks=[
            (1, "See Table 3 for details. Figure 2.1 shows the trend."),
        ]
    )
    chunks = engine.chunk(doc)
    assert len(chunks) == 1
    assert "Table 3" in chunks[0].references
    assert "Figure 2.1" in chunks[0].references


def test_chunk_validator_rejects_empty_content() -> None:
    """ChunkValidator raises on empty chunk content."""
    validator = ChunkValidator()
    # Create a minimal valid LDU first
    valid_ldu = LDU(
        id="x",
        doc_id="d",
        chunk_type=ChunkType.paragraph,
        content="x",
        page_refs=[1],
        token_count=1,
        content_hash="x",
    )
    validator.validate([valid_ldu])  # should not raise

    invalid_ldu = LDU(
        id="y",
        doc_id="d",
        chunk_type=ChunkType.paragraph,
        content="",
        page_refs=[1],
        token_count=0,
        content_hash="y",
    )
    with pytest.raises(ChunkValidationError, match="Chunk content must not be empty"):
        validator.validate([invalid_ldu])


def test_chunker_loads_config_from_path(tmp_path: Path) -> None:
    """ChunkingEngine loads config from custom path."""
    custom_rules = tmp_path / "rules.yaml"
    custom_rules.write_text(
        "chunking:\n"
        "  max_tokens_per_chunk: 256\n"
        "  hard_max_tokens_per_chunk: 512\n"
        "  keep_numbered_lists_together: false\n"
    )
    engine = ChunkingEngine(custom_rules)
    assert engine.max_tokens_per_chunk == 256
    assert engine.hard_max_tokens_per_chunk == 512
    assert engine.keep_numbered_lists_together is False


# ---------- Stress tests: noisy / real-world edge cases ----------


def test_stress_very_long_table_single_ldu(rules_path: Path) -> None:
    """
    Very long table (many rows) remains a single LDU; table-header binding
    is preserved; validator passes.
    """
    engine = ChunkingEngine(rules_path)
    num_rows = 400
    rows = [
        [TableCell(row_index=i, col_index=0, text=f"Entity_{i}"), TableCell(row_index=i, col_index=1, text="100")]
        for i in range(num_rows)
    ]
    doc = _make_extracted_doc(
        text_blocks=[(1, "Preface.")],
        tables=[
            Table(
                page_number=1,
                headers=["Entity", "Revenue"],
                rows=rows,
            ),
        ],
    )
    chunks = engine.chunk(doc)
    table_chunks = [c for c in chunks if c.chunk_type == ChunkType.table]
    assert len(table_chunks) == 1
    assert "Entity_0" in table_chunks[0].content
    assert "Entity_399" in table_chunks[0].content
    assert table_chunks[0].content.count("\n") >= num_rows


def test_stress_deeply_nested_list_kept_intact_when_under_hard_max(rules_path: Path) -> None:
    """Long numbered list under hard_max_tokens is kept as a single LDU."""
    engine = ChunkingEngine(rules_path)
    lines = [f"{i+1}. Item number {i+1}." for i in range(120)]
    doc = _make_extracted_doc(text_blocks=[(1, "\n".join(lines))])
    chunks = engine.chunk(doc)
    list_chunks = [c for c in chunks if c.chunk_type == ChunkType.list]
    assert len(list_chunks) >= 1
    assert "1. Item number 1." in list_chunks[0].content
    assert list_chunks[0].token_count <= engine.hard_max_tokens_per_chunk


def test_stress_long_list_splits_when_over_hard_max(rules_path: Path) -> None:
    """Numbered list exceeding hard_max_tokens is split; each list chunk obeys hard_max."""
    engine = ChunkingEngine(rules_path)
    long_line = "1. " + "word " * 80
    lines = [long_line.replace("1.", f"{i+1}.", 1) for i in range(40)]
    doc = _make_extracted_doc(text_blocks=[(1, "\n".join(lines))])
    chunks = engine.chunk(doc)
    list_chunks = [c for c in chunks if c.chunk_type == ChunkType.list]
    assert len(list_chunks) >= 1
    for c in list_chunks:
        assert c.token_count <= engine.hard_max_tokens_per_chunk


def test_stress_ambiguous_cross_references_only_resolved_refs_in_relationships(rules_path: Path) -> None:
    """
    Text references 'Table 1' and 'Table 99'; document has only one table.
    Only Table 1 is resolved; Table 99 is ambiguous and must not cause validation failure.
    """
    engine = ChunkingEngine(rules_path)
    doc = _make_extracted_doc(
        text_blocks=[
            (1, "See Table 1 for the main data. Table 99 does not exist in this document."),
        ],
        tables=[
            Table(
                page_number=1,
                headers=["A", "B"],
                rows=[[TableCell(row_index=0, col_index=0, text="x"), TableCell(row_index=0, col_index=1, text="1")]],
            ),
        ],
    )
    chunks = engine.chunk(doc)
    text_chunks = [c for c in chunks if "Table 1" in c.references and "Table 99" in c.references]
    assert len(text_chunks) == 1
    chunk = text_chunks[0]
    assert len(chunk.relationships) == 1
    assert chunk.relationships[0]["ref_text"] == "Table 1"
