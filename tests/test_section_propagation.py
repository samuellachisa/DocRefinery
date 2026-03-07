"""
Tests for section header propagation and PageIndex integration.
Verifies that parent_section metadata is set correctly across complex
hierarchies when enrich_ldus_with_sections is applied.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.chunker import ChunkingEngine
from src.agents.indexer import enrich_ldus_with_sections, _section_for_page
from src.models import BoundingBox, ExtractedDocument, LDU, TextBlock
from src.models.ldu import ChunkType
from src.models.page_index import PageIndex, SectionNode


def _make_ldu(ldu_id: str, doc_id: str, page_refs: list[int], content: str = "x") -> LDU:
    return LDU(
        id=ldu_id,
        doc_id=doc_id,
        chunk_type=ChunkType.paragraph,
        content=content,
        page_refs=page_refs,
        token_count=1,
        content_hash=ldu_id,
    )


def test_section_for_page_returns_deepest_section() -> None:
    """_section_for_page returns the deepest section node that covers the page."""
    child_2_1 = SectionNode(
        id="sec-2.1",
        title="Subsection 2.1",
        page_start=3,
        page_end=4,
        children=[],
    )
    child_2_2 = SectionNode(
        id="sec-2.2",
        title="Subsection 2.2",
        page_start=5,
        page_end=5,
        children=[],
    )
    child_2 = SectionNode(
        id="sec-2",
        title="Section 2",
        page_start=3,
        page_end=5,
        children=[child_2_1, child_2_2],
    )
    child_1 = SectionNode(
        id="sec-1",
        title="Section 1",
        page_start=1,
        page_end=2,
        children=[],
    )
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=5,
        children=[child_1, child_2],
    )

    assert _section_for_page(root, 1) is child_1
    assert _section_for_page(root, 2) is child_1
    assert _section_for_page(root, 3) is child_2_1
    assert _section_for_page(root, 4) is child_2_1
    assert _section_for_page(root, 5) is child_2_2
    assert _section_for_page(root, 0) is None
    assert _section_for_page(root, 6) is None


def test_enrich_ldus_sets_parent_section_from_complex_hierarchy() -> None:
    """
    enrich_ldus_with_sections sets parent_section to the title of the
    deepest section covering each LDU's first page.
    """
    doc_id = "hier-doc"
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=5,
        children=[
            SectionNode(id="s1", title="Executive Summary", page_start=1, page_end=2, children=[]),
            SectionNode(
                id="s2",
                title="Methods",
                page_start=3,
                page_end=5,
                children=[
                    SectionNode(id="s2a", title="2.1 Data Collection", page_start=3, page_end=4, children=[]),
                    SectionNode(id="s2b", title="2.2 Analysis", page_start=5, page_end=5, children=[]),
                ],
            ),
        ],
    )
    page_index = PageIndex(doc_id=doc_id, root=root)

    ldus = [
        _make_ldu("l1", doc_id, [1]),
        _make_ldu("l2", doc_id, [2]),
        _make_ldu("l3", doc_id, [3]),
        _make_ldu("l4", doc_id, [4]),
        _make_ldu("l5", doc_id, [5]),
    ]
    enrich_ldus_with_sections(ldus, page_index)

    assert ldus[0].parent_section == "Executive Summary"
    assert ldus[1].parent_section == "Executive Summary"
    assert ldus[2].parent_section == "2.1 Data Collection"
    assert ldus[3].parent_section == "2.1 Data Collection"
    assert ldus[4].parent_section == "2.2 Analysis"


def test_enrich_ldus_skips_when_page_outside_index_range() -> None:
    """When an LDU's page is outside the PageIndex range, parent_section is not set (no crash)."""
    root = SectionNode(id="r", title="Doc", page_start=1, page_end=2, children=[])
    page_index = PageIndex(doc_id="d", root=root)
    ldu = _make_ldu("x", "d", [99])  # page 99 not in index
    enrich_ldus_with_sections([ldu], page_index)
    assert ldu.parent_section is None


def test_section_propagation_integration_chunker_then_enrich(rules_path: Path) -> None:
    """
    Integration: chunk document -> apply a concrete PageIndex hierarchy ->
    enrich_ldus_with_sections -> parent_section verifiably correct for each LDU.
    Uses a manually built PageIndex so the test does not depend on LLM.
    """
    doc = ExtractedDocument(
        doc_id="int-doc",
        num_pages=3,
        text_blocks=[
            TextBlock(
                page_number=1,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                text="Introduction paragraph.",
                reading_order=0,
            ),
            TextBlock(
                page_number=2,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                text="Methods paragraph.",
                reading_order=1,
            ),
            TextBlock(
                page_number=3,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=100),
                text="Results paragraph.",
                reading_order=2,
            ),
        ],
        tables=[],
        figures=[],
    )
    chunker = ChunkingEngine(rules_path)
    ldus = chunker.chunk(doc)
    assert len(ldus) == 3

    # Manually build a hierarchy: root -> Page 1, Page 2, Page 3 (as in flat index)
    page_index = PageIndex(
        doc_id="int-doc",
        root=SectionNode(
            id="root",
            title="Document",
            page_start=1,
            page_end=3,
            children=[
                SectionNode(id="p1", title="Page 1", page_start=1, page_end=1, children=[]),
                SectionNode(id="p2", title="Page 2", page_start=2, page_end=2, children=[]),
                SectionNode(id="p3", title="Page 3", page_start=3, page_end=3, children=[]),
            ],
        ),
    )
    enrich_ldus_with_sections(ldus, page_index)

    for ldu in ldus:
        assert ldu.page_refs
        page = ldu.page_refs[0]
        assert ldu.parent_section is not None
        section = _section_for_page(page_index.root, page)
        assert section is not None
        assert ldu.parent_section == section.title, (
            f"LDU on page {page}: expected parent_section {section.title}, got {ldu.parent_section}"
        )
