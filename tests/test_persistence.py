"""
Tests for data persistence: round-trip integrity between LDUs, content_hashes,
and fact rows; vector store metadata; multi-period extraction.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.agents.chunker import ChunkingEngine
from src.agents.fact_extractor import FactExtractor
from src.agents.query_agent import FactTable, VectorStore, VectorStoreConfig
from src.models import BoundingBox, ExtractedDocument, LDU, Table, TableCell
from src.models.ldu import ChunkType
from src.utils.content_hash import content_hash


def _make_table(
    page_number: int = 1,
    headers: list[str] | None = None,
    rows: list[list[str]] | None = None,
) -> Table:
    if headers is None:
        headers = ["Entity", "Revenue", "Cost"]
    if rows is None:
        rows = [
            ["Division A", "100", "60"],
            ["Division B", "200", "120"],
        ]
    table_rows = [
        [
            TableCell(row_index=i, col_index=j, text=cell_text)
            for j, cell_text in enumerate(row)
        ]
        for i, row in enumerate(rows)
    ]
    return Table(
        page_number=page_number,
        headers=headers,
        rows=table_rows,
    )


def _table_content_for_hash(table: Table) -> str:
    """Same format as chunker and fact_extractor for consistent content_hash."""
    header_line = " | ".join(table.headers) if table.headers else ""
    row_lines = [" | ".join(cell.text for cell in row) for row in table.rows]
    return "\n".join([header_line] + row_lines if header_line else row_lines).strip()


@pytest.fixture
def rules_path(tmp_path: Path) -> Path:
    r = tmp_path / "rules.yaml"
    r.write_text(
        "chunking:\n"
        "  max_tokens_per_chunk: 512\n"
        "  hard_max_tokens_per_chunk: 1024\n"
        "  keep_numbered_lists_together: true\n"
    )
    return r


@pytest.fixture
def temp_fact_db(tmp_path: Path) -> Path:
    return tmp_path / "facts.sqlite"


@pytest.fixture
def temp_chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


def test_fact_content_hash_matches_ldu(
    rules_path: Path, temp_fact_db: Path
) -> None:
    """
    Round-trip: table -> LDU (chunker) -> fact extraction.
    Every fact row from a table must have content_hash equal to that table LDU's content_hash.
    """
    doc_id = "doc-1"
    table = _make_table(page_number=1)
    doc = ExtractedDocument(
        doc_id=doc_id,
        num_pages=1,
        text_blocks=[],
        tables=[table],
        figures=[],
    )
    chunker = ChunkingEngine(rules_path)
    ldus = chunker.chunk(doc)
    table_ldus = [u for u in ldus if u.chunk_type == ChunkType.table]
    assert len(table_ldus) == 1
    expected_hash = table_ldus[0].content_hash

    fact_table = FactTable(temp_fact_db)
    FactExtractor().extract(doc, fact_table)
    rows = fact_table.query_by_content_hash(expected_hash)
    assert len(rows) >= 1
    for r in rows:
        assert r["content_hash"] == expected_hash
        assert r["doc_id"] == doc_id
        assert r["page_number"] == 1


def test_vector_metadata_content_hash_matches_ldu(
    rules_path: Path, temp_chroma_dir: Path
) -> None:
    """
    Ingest LDUs into vector store; verify stored metadata content_hash matches LDU content_hash.
    """
    doc = ExtractedDocument(
        doc_id="vec-doc",
        num_pages=1,
        text_blocks=[],
        tables=[_make_table()],
        figures=[],
    )
    chunker = ChunkingEngine(rules_path)
    ldus = chunker.chunk(doc)
    assert ldus

    store = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma_dir)))
    store.ingest(ldus)

    for ldu in ldus:
        # Chroma get by id
        res = store.collection.get(ids=[ldu.id], include=["metadatas"])
        metas = res.get("metadatas", [[]])
        assert metas, f"LDU {ldu.id} not found in collection"
        meta = metas[0]
        assert meta.get("content_hash") == ldu.content_hash
        assert meta.get("doc_id") == ldu.doc_id


def test_round_trip_integrity_ldu_to_facts(
    rules_path: Path, temp_fact_db: Path
) -> None:
    """
    Full round-trip: ExtractedDocument with tables -> chunk to LDUs -> extract facts.
    Assert every fact row has content_hash and page_number consistent with the source table LDU.
    """
    doc_id = "roundtrip-doc"
    table1 = _make_table(page_number=1, headers=["Item", "Q1", "Q2"], rows=[["X", "10", "20"], ["Y", "30", "40"]])
    doc = ExtractedDocument(
        doc_id=doc_id,
        num_pages=2,
        text_blocks=[],
        tables=[table1],
        figures=[],
    )
    chunker = ChunkingEngine(rules_path)
    ldus = chunker.chunk(doc)
    table_ldus = {u.content_hash: u for u in ldus if u.chunk_type == ChunkType.table}
    assert len(table_ldus) == 1
    ldu_hash = list(table_ldus.keys())[0]
    ldu = table_ldus[ldu_hash]

    fact_table = FactTable(temp_fact_db)
    FactExtractor().extract(doc, fact_table)
    facts = fact_table.query_by_content_hash(ldu_hash)
    assert len(facts) >= 2
    for f in facts:
        assert f["content_hash"] == ldu.content_hash
        assert f["page_number"] == ldu.page_refs[0]
        assert f["doc_id"] == doc_id


def test_multi_period_extraction(temp_fact_db: Path) -> None:
    """
    Multi-period table: headers like Entity, 2022, 2023, 2024.
    Expect one fact per cell with period 2022/2023/2024 and base metric from title.
    """
    doc_id = "multi-period-doc"
    table = Table(
        page_number=1,
        title="Revenue",
        headers=["Entity", "2022", "2023", "2024"],
        rows=[
            [
                TableCell(row_index=0, col_index=0, text="Region A"),
                TableCell(row_index=0, col_index=1, text="100"),
                TableCell(row_index=0, col_index=2, text="110"),
                TableCell(row_index=0, col_index=3, text="120"),
            ],
            [
                TableCell(row_index=1, col_index=0, text="Region B"),
                TableCell(row_index=1, col_index=1, text="200"),
                TableCell(row_index=1, col_index=2, text="220"),
                TableCell(row_index=1, col_index=3, text="240"),
            ],
        ],
    )
    doc = ExtractedDocument(doc_id=doc_id, num_pages=1, text_blocks=[], tables=[table], figures=[])
    fact_table = FactTable(temp_fact_db)
    FactExtractor().extract(doc, fact_table)

    by_entity = fact_table.query_by_entity(doc_id, "Region A")
    assert len(by_entity) >= 3
    periods = {r["period"] for r in by_entity}
    assert "2022" in periods
    assert "2023" in periods
    assert "2024" in periods

    in_range = fact_table.query_by_period_range(doc_id, period_start="2022", period_end="2023")
    assert len(in_range) >= 4


def test_query_metric_with_limit_offset(temp_fact_db: Path) -> None:
    """Optimized query_metric with limit/offset returns correct window."""
    doc_id = "limit-doc"
    table = _make_table(headers=["E", "Revenue"], rows=[[f"Ent{i}", str(100 + i)] for i in range(10)])
    doc = ExtractedDocument(doc_id=doc_id, num_pages=1, text_blocks=[], tables=[table], figures=[])
    fact_table = FactTable(temp_fact_db)
    FactExtractor().extract(doc, fact_table)

    all_rows = fact_table.query_metric(doc_id, "revenue")
    first_two = fact_table.query_metric(doc_id, "revenue", limit=2, offset=0)
    next_two = fact_table.query_metric(doc_id, "revenue", limit=2, offset=2)
    assert len(first_two) == 2
    assert len(next_two) == 2
    assert first_two[0]["id"] != next_two[0]["id"]


def test_ingest_batch_preserves_metadata(
    rules_path: Path, temp_chroma_dir: Path
) -> None:
    """Batch ingest preserves content_hash and doc_id in metadata."""
    doc = ExtractedDocument(
        doc_id="batch-doc",
        num_pages=1,
        text_blocks=[],
        tables=[_make_table()],
        figures=[],
    )
    chunker = ChunkingEngine(rules_path)
    ldus = chunker.chunk(doc)
    store = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma_dir)))
    store.ingest_batch(ldus, batch_size=2)
    for ldu in ldus:
        res = store.collection.get(ids=[ldu.id], include=["metadatas"])
        assert res["metadatas"][0]["content_hash"] == ldu.content_hash
