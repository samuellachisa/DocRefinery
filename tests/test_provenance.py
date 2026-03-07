"""
Tests for the provenance pipeline: spans with bbox/content_hash, chains attached to answers,
audit_claim with support for low-confidence and contradictory evidence, and multi-page spans.
Ensures provenance is correctly populated across extraction-derived content (paragraph, table, figure).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.chunker import ChunkingEngine
from src.agents.query_agent import (
    AuditResult,
    QueryAgent,
    VectorStore,
    VectorStoreConfig,
    FactTable,
    _span_from_ldu,
)
from src.models import (
    BoundingBox,
    ExtractedDocument,
    Figure,
    LDU,
    Table,
    TableCell,
    TextBlock,
)
from src.models.ldu import ChunkType
from src.models.page_index import PageIndex, SectionNode
from src.models.provenance import ProvenanceChain, ProvenanceSpan


def _minimal_page_index(doc_id: str = "test-doc") -> PageIndex:
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=2,
        summary="Root section.",
    )
    return PageIndex(doc_id=doc_id, root=root)


def _ldu_paragraph(doc_id: str, page: int = 1, content: str = "Paragraph content.") -> LDU:
    return LDU(
        id=f"p-{doc_id}-{page}",
        doc_id=doc_id,
        chunk_type=ChunkType.paragraph,
        content=content,
        page_refs=[page],
        bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=100),
        token_count=10,
        content_hash=f"hash-p-{page}",
    )


def _ldu_multi_page(doc_id: str, pages: list[int], content: str = "Spans pages.") -> LDU:
    return LDU(
        id=f"mp-{doc_id}-{min(pages)}-{max(pages)}",
        doc_id=doc_id,
        chunk_type=ChunkType.paragraph,
        content=content,
        page_refs=pages,
        bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=100),
        token_count=10,
        content_hash=f"hash-mp-{min(pages)}-{max(pages)}",
    )


@pytest.fixture
def temp_chroma(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def temp_fact_db(tmp_path: Path) -> Path:
    return tmp_path / "facts.sqlite"


@pytest.fixture
def query_agent(temp_chroma: Path, temp_fact_db: Path) -> QueryAgent:
    page_index = _minimal_page_index()
    ldus = [_ldu_paragraph(page_index.doc_id, 1)]
    vs = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma)))
    ft = FactTable(db_path=temp_fact_db)
    return QueryAgent(page_index=page_index, ldus=ldus, vector_store=vs, fact_table=ft)


# ---------- Provenance span structure (bbox, content_hash, multi-page) ----------


def test_span_from_ldu_single_page_has_required_fields() -> None:
    """ProvenanceSpan from single-page LDU has doc_id, page_number, content_hash or bbox, snippet."""
    ldu = _ldu_paragraph("doc1", 2)
    span = _span_from_ldu(ldu, "report.pdf")
    assert span.doc_id == "doc1"
    assert span.doc_name == "report.pdf"
    assert span.page_number == 2
    assert span.page_end is None
    assert span.content_hash == "hash-p-2"
    assert span.snippet is not None
    assert span.bbox is not None


def test_span_from_ldu_multi_page_sets_page_end() -> None:
    """ProvenanceSpan from multi-page LDU has page_number and page_end."""
    ldu = _ldu_multi_page("doc1", [2, 3], "Content spanning two pages.")
    span = _span_from_ldu(ldu, "report.pdf")
    assert span.page_number == 2
    assert span.page_end == 3
    assert span.snippet == "Content spanning two pages."


def test_answer_provenance_multi_page_span(query_agent: QueryAgent) -> None:
    """Answer built from semantic search returns provenance with page_end when LDU spans multiple pages."""
    multi = _ldu_multi_page(query_agent.page_index.doc_id, [2, 3], "Multi-page answer.")
    query_agent.ldus[multi.id] = multi
    query_agent.vector_store.ingest([multi])
    with patch.object(query_agent, "semantic_search", return_value=[multi]):
        text, chain = query_agent.answer("multi-page?", "doc.pdf")
    assert len(chain.spans) == 1
    assert chain.spans[0].page_number == 2
    assert chain.spans[0].page_end == 3


# ---------- Provenance populated for extraction-derived content (paragraph, table, figure) ----------


def test_provenance_populated_after_chunking_paragraph_content(
    temp_chroma: Path, temp_fact_db: Path
) -> None:
    """Provenance is correctly populated when answer comes from paragraph-derived LDU (e.g. fast_text)."""
    doc = ExtractedDocument(
        doc_id="ext-doc",
        num_pages=1,
        text_blocks=[
            TextBlock(
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                text="The revenue increased in 2023.",
                reading_order=0,
            )
        ],
        tables=[],
        figures=[],
    )
    chunker = ChunkingEngine()
    ldus = chunker.chunk(doc)
    assert len(ldus) >= 1
    page_index = _minimal_page_index("ext-doc")
    vs = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma)))
    ft = FactTable(db_path=temp_fact_db)
    agent = QueryAgent(page_index=page_index, ldus=ldus, vector_store=vs, fact_table=ft)
    text, chain = agent.answer("What happened to revenue?", "extracted.pdf")
    assert chain.spans
    for span in chain.spans:
        assert span.doc_id == "ext-doc"
        assert span.page_number >= 1
        assert span.content_hash or span.bbox
        assert span.snippet


def test_provenance_populated_after_chunking_table_content(
    temp_chroma: Path, temp_fact_db: Path
) -> None:
    """Provenance is correctly populated when content comes from table-derived LDU (e.g. layout extraction)."""
    doc = ExtractedDocument(
        doc_id="table-doc",
        num_pages=1,
        text_blocks=[],
        tables=[
            Table(
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=200, y1=80),
                headers=["Metric", "Value"],
                rows=[
                    [
                        TableCell(row_index=0, col_index=0, text="Revenue"),
                        TableCell(row_index=0, col_index=1, text="100"),
                    ]
                ],
                title=None,
                caption=None,
            )
        ],
        figures=[],
    )
    chunker = ChunkingEngine()
    ldus = chunker.chunk(doc)
    assert len(ldus) >= 1
    page_index = _minimal_page_index("table-doc")
    vs = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma)))
    ft = FactTable(db_path=temp_fact_db)
    agent = QueryAgent(page_index=page_index, ldus=ldus, vector_store=vs, fact_table=ft)
    text, chain = agent.answer("What is the revenue?", "tables.pdf")
    assert chain.spans
    for span in chain.spans:
        assert span.doc_id == "table-doc"
        assert span.page_number == 1
        assert span.content_hash or span.bbox
        assert span.snippet


def test_provenance_populated_after_chunking_figure_caption(
    temp_chroma: Path, temp_fact_db: Path
) -> None:
    """Provenance is correctly populated when content comes from figure caption LDU (e.g. vision extraction)."""
    doc = ExtractedDocument(
        doc_id="figure-doc",
        num_pages=1,
        text_blocks=[],
        tables=[],
        figures=[
            Figure(
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=150, y1=150),
                caption="Chart shows revenue growth in Q3.",
                label="Figure 1",
            )
        ],
    )
    chunker = ChunkingEngine()
    ldus = chunker.chunk(doc)
    assert len(ldus) >= 1
    page_index = _minimal_page_index("figure-doc")
    vs = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma)))
    ft = FactTable(db_path=temp_fact_db)
    agent = QueryAgent(page_index=page_index, ldus=ldus, vector_store=vs, fact_table=ft)
    text, chain = agent.answer("What does the chart show?", "figures.pdf")
    assert chain.spans
    for span in chain.spans:
        assert span.doc_id == "figure-doc"
        assert span.page_number == 1
        assert span.content_hash or span.bbox
        assert span.snippet


def test_provenance_chain_serializable() -> None:
    """ProvenanceChain with multi-page span serializes to JSON-friendly dict."""
    span = ProvenanceSpan(
        doc_name="doc.pdf",
        doc_id="id1",
        page_number=2,
        page_end=4,
        content_hash="h1",
        snippet="Snippet.",
    )
    chain = ProvenanceChain(spans=[span])
    d = chain.model_dump(mode="json")
    assert d["spans"][0]["page_number"] == 2
    assert d["spans"][0]["page_end"] == 4


# ---------- Audit mode: unverifiable, verified, low-confidence, contradiction ----------


def test_audit_claim_unverifiable_returns_audit_result(query_agent: QueryAgent) -> None:
    """When no supporting content is found, audit_claim returns verified=False and confidence=none."""
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[],
    ):
        result = query_agent.audit_claim("Something unverifiable.", "doc.pdf")
    assert isinstance(result, AuditResult)
    assert result.verified is False
    assert result.confidence == "none"
    assert result.contradiction_detected is False
    assert len(result.provenance.spans) == 0
    assert "Flagged as unverifiable" in result.supporting_text


def test_audit_claim_verified_high_confidence(query_agent: QueryAgent) -> None:
    """When best match has low distance, confidence is high."""
    ldu = _ldu_paragraph(query_agent.page_index.doc_id, 1, "Revenue was 100M.")
    query_agent.ldus[ldu.id] = ldu
    # Chroma returns L2 distance; 0.3 is "close"
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[(ldu.id, {"content": ldu.content}, 0.3)],
    ):
        result = query_agent.audit_claim("Revenue was 100M.", "doc.pdf")
    assert result.verified is True
    assert result.confidence == "high"
    assert len(result.provenance.spans) == 1


def test_audit_claim_low_confidence_when_distance_above_threshold(
    query_agent: QueryAgent,
) -> None:
    """When best match distance is above threshold, confidence is low."""
    ldu = _ldu_paragraph(query_agent.page_index.doc_id, 1, "Some text.")
    query_agent.ldus[ldu.id] = ldu
    # Distance 2.0 > default threshold 1.0
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[(ldu.id, {"content": ldu.content}, 2.0)],
    ):
        result = query_agent.audit_claim("Some claim.", "doc.pdf")
    assert result.verified is True
    assert result.confidence == "low"


def test_audit_claim_contradiction_detected_when_phrase_present(
    query_agent: QueryAgent,
) -> None:
    """When a retrieved snippet contains a contradiction phrase, contradiction_detected is True."""
    ldu = _ldu_paragraph(
        query_agent.page_index.doc_id,
        1,
        "This result is contrary to the initial hypothesis.",
    )
    query_agent.ldus[ldu.id] = ldu
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[(ldu.id, {"content": ldu.content}, 0.2)],
    ):
        result = query_agent.audit_claim("Initial hypothesis.", "doc.pdf")
    assert result.verified is True
    assert result.contradiction_detected is True


def test_audit_claim_no_contradiction_when_phrase_absent(query_agent: QueryAgent) -> None:
    """When no snippet contains contradiction phrase, contradiction_detected is False."""
    ldu = _ldu_paragraph(
        query_agent.page_index.doc_id,
        1,
        "This result supports the claim.",
    )
    query_agent.ldus[ldu.id] = ldu
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[(ldu.id, {"content": ldu.content}, 0.2)],
    ):
        result = query_agent.audit_claim("The claim.", "doc.pdf")
    assert result.verified is True
    assert result.contradiction_detected is False


def test_audit_claim_provenance_multi_page_spans(query_agent: QueryAgent) -> None:
    """Audit result provenance includes page_end for multi-page LDUs."""
    multi = _ldu_multi_page(query_agent.page_index.doc_id, [1, 2], "Evidence across pages.")
    query_agent.ldus[multi.id] = multi
    with patch.object(
        query_agent.vector_store,
        "query_with_distances",
        return_value=[(multi.id, {"content": multi.content}, 0.1)],
    ):
        result = query_agent.audit_claim("Evidence.", "doc.pdf")
    assert len(result.provenance.spans) == 1
    assert result.provenance.spans[0].page_number == 1
    assert result.provenance.spans[0].page_end == 2
