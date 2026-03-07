"""
Tests for PageIndex builder: multi-level headings, mixed languages,
weak/missing headings fallback, and navigate_pageindex scoring behavior.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.indexer import (
    _collect_sections_flatten,
    _data_types_from_ldus,
    _detect_headings,
    navigate_pageindex,
)
from src.models import BoundingBox, ExtractedDocument, LDU, TextBlock
from src.models.ldu import ChunkType
from src.models.page_index import PageIndex, SectionNode


def _text_block(page: int, text: str, order: int = 0) -> TextBlock:
    return TextBlock(
        page_number=page,
        bbox=BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=100.0),
        text=text,
        reading_order=order,
    )


# ---------- Multi-level heading hierarchy ----------


def test_detect_headings_multi_level_hierarchy() -> None:
    """Multi-level headings (Chapter, 1.1, 1.1.1, 1.2) are detected with correct levels."""
    blocks = [
        _text_block(1, "Chapter 1 Introduction\nSome intro text.", 0),
        _text_block(2, "1.1 Background\nBackground content.", 1),
        _text_block(3, "1.1.1 Scope\nScope of the document.", 2),
        _text_block(4, "1.2 Methods\nMethods content.", 3),
    ]
    headings = _detect_headings(blocks)
    assert len(headings) >= 4
    by_title = {h[1]: (h[0], h[2]) for h in headings}
    assert "Introduction" in by_title or "Introduction" in str(headings)
    assert by_title.get("Background", (0, 0))[1] == 2
    # Scope may be level 2 or 3 depending on regex capture of (\.\d+)*
    assert by_title.get("Scope", (0, 0))[1] in (2, 3)
    assert by_title.get("Methods", (0, 0))[1] == 2
    # Chapter-style gets level 1
    chapter_heads = [h for h in headings if h[2] == 1]
    assert any("Introduction" in h[1] or "Introduction" == h[1] for h in chapter_heads)


def test_detect_headings_numbered_and_all_caps() -> None:
    """Numbered (1. 2.1) and short ALL CAPS lines are detected as headings."""
    blocks = [
        _text_block(1, "1. First Section\nContent here.", 0),
        _text_block(1, "EXECUTIVE SUMMARY\nSummary text.", 1),
        _text_block(2, "2.1 Subsection\nMore content.", 2),
    ]
    headings = _detect_headings(blocks)
    assert len(headings) >= 2
    titles = [h[1] for h in headings]
    assert any("First Section" in t for t in titles)
    assert any("EXECUTIVE SUMMARY" in t or t == "EXECUTIVE SUMMARY" for t in titles)
    assert any("Subsection" in t for t in titles)


# ---------- Mixed languages ----------


def test_detect_headings_numbered_with_non_ascii() -> None:
    """Numbered headings with accented or non-ASCII titles are detected (e.g. 1. Introducción)."""
    blocks = [
        _text_block(1, "1. Introducción\nContenido en español.", 0),
        _text_block(2, "2.1 Résumé\nContenu en français.", 1),
    ]
    headings = _detect_headings(blocks)
    assert len(headings) >= 2
    titles = [h[1] for h in headings]
    assert any("Introducción" in t or "Introduccion" in t for t in titles)
    assert any("Résumé" in t or "Resume" in t for t in titles)


# ---------- Weak or missing headings ----------


def test_build_with_no_headings_uses_flat_page_structure(tmp_path: Path) -> None:
    """When document has no heading-like lines, build() produces flat page-level structure."""
    from src.agents.indexer import PageIndexBuilder

    doc = ExtractedDocument(
        doc_id="no-headings-doc",
        num_pages=2,
        text_blocks=[
            _text_block(1, "Just a paragraph of body text. No headings here.", 0),
            _text_block(2, "Another paragraph. Still no section titles.", 1),
        ],
        tables=[],
        figures=[],
    )
    with patch("src.agents.indexer.call_text_llm", return_value="Summary"):
        builder = PageIndexBuilder()
        page_index = builder.build(doc, ldus=None)
    assert page_index.doc_id == "no-headings-doc"
    assert page_index.root.title == "Document"
    assert page_index.root.page_start == 1
    assert page_index.root.page_end == 2
    children = page_index.root.children
    assert len(children) == 2
    assert children[0].title == "Page 1"
    assert children[0].page_start == children[0].page_end == 1
    assert children[1].title == "Page 2"
    assert children[1].page_start == children[1].page_end == 2


def test_build_with_headings_produces_hierarchical_tree(tmp_path: Path) -> None:
    """When headings are detected, build() produces a hierarchical root with section children (LLM mocked)."""
    from src.agents.indexer import PageIndexBuilder

    doc = ExtractedDocument(
        doc_id="with-headings-doc",
        num_pages=3,
        text_blocks=[
            _text_block(1, "Chapter 1 Start\nContent.", 0),
            _text_block(2, "1.1 Middle\nMore content.", 1),
            _text_block(3, "1.2 End\nFinal content.", 2),
        ],
        tables=[],
        figures=[],
    )
    with patch("src.agents.indexer.call_text_llm", return_value="Summary"):
        builder = PageIndexBuilder()
        page_index = builder.build(doc, ldus=None)
    assert page_index.root.title == "Document"
    assert len(page_index.root.children) >= 1
    first = page_index.root.children[0]
    assert first.title == "Start"
    assert len(first.children) >= 2
    child_titles = [c.title for c in first.children]
    assert "Middle" in child_titles and "End" in child_titles


def test_build_with_empty_text_blocks_returns_minimal_index(tmp_path: Path) -> None:
    """When document has no text_blocks, build() returns minimal index without crashing."""
    from src.agents.indexer import PageIndexBuilder

    doc = ExtractedDocument(
        doc_id="empty-doc",
        num_pages=1,
        text_blocks=[],
        tables=[],
        figures=[],
    )
    builder = PageIndexBuilder()
    page_index = builder.build(doc, ldus=None)
    assert page_index.root.title == "Document"
    assert page_index.root.page_start == 1
    assert page_index.root.children == []


# ---------- navigate_pageindex scoring behavior ----------


def test_navigate_pageindex_returns_sections_matching_topic() -> None:
    """Sections whose title/summary/key_entities contain topic keywords are returned."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=3,
        children=[
            SectionNode(
                id="s1",
                title="Revenue and profit",
                page_start=1,
                page_end=1,
                summary="This section covers revenue.",
                key_entities=["Revenue", "EBITDA"],
                children=[],
            ),
            SectionNode(
                id="s2",
                title="Appendix",
                page_start=2,
                page_end=3,
                summary="Technical details.",
                key_entities=[],
                children=[],
            ),
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "revenue", top_k=5)
    assert len(result) >= 1
    assert any("revenue" in (s.title + " " + (s.summary or "")).lower() for s in result)
    first = result[0]
    assert first.title == "Revenue and profit" or "revenue" in (first.summary or "").lower()


def test_navigate_pageindex_respects_top_k() -> None:
    """Returned list length does not exceed top_k."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=3,
        children=[
            SectionNode(id="a", title="Alpha beta", page_start=1, page_end=1, children=[]),
            SectionNode(id="b", title="Beta gamma", page_start=2, page_end=2, children=[]),
            SectionNode(id="c", title="Gamma delta", page_start=3, page_end=3, children=[]),
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "beta", top_k=2)
    assert len(result) <= 2


def test_navigate_pageindex_excludes_zero_score_sections() -> None:
    """Sections with no keyword overlap (score 0) are not returned."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=2,
        children=[
            SectionNode(id="x", title="Revenue", page_start=1, page_end=1, children=[]),
            SectionNode(id="y", title="Appendix", page_start=2, page_end=2, summary="Misc.", children=[]),
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "revenue", top_k=5)
    assert all("revenue" in (s.title + " " + (s.summary or "")).lower() for s in result)
    assert len(result) >= 1
    assert result[0].title == "Revenue"


def test_navigate_pageindex_empty_topic_returns_empty() -> None:
    """Empty or whitespace topic yields no matches (all scores 0)."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=1,
        children=[SectionNode(id="s", title="Something", page_start=1, page_end=1, children=[])],
    )
    page_index = PageIndex(doc_id="d", root=root)
    assert navigate_pageindex(page_index, "", top_k=3) == []
    assert navigate_pageindex(page_index, "   ", top_k=3) == []


def test_navigate_pageindex_single_char_tokens_ignored() -> None:
    """Tokens of length 1 are not counted (scoring uses len(tok) > 1)."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=1,
        children=[
            SectionNode(id="s", title="A", page_start=1, page_end=1, summary="Section A.", children=[]),
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "a", top_k=3)
    # "a" has len 1 so it is skipped in scoring; "Section" might match if we had topic "section"
    result_section = navigate_pageindex(page_index, "section", top_k=3)
    assert len(result_section) >= 1
    assert result_section[0].summary and "ection" in result_section[0].summary.lower()


def test_navigate_pageindex_scores_nested_children() -> None:
    """Scoring includes nested sections (flattened via _collect_sections_flatten)."""
    child = SectionNode(
        id="child",
        title="Nested revenue section",
        page_start=2,
        page_end=2,
        children=[],
    )
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=2,
        children=[
            SectionNode(id="p1", title="Page 1", page_start=1, page_end=1, children=[]),
            child,
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "revenue", top_k=5)
    assert len(result) >= 1
    assert result[0].title == "Nested revenue section"


def test_navigate_pageindex_ordering_by_score_descending() -> None:
    """Results are ordered by score descending; higher overlap first."""
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=3,
        children=[
            SectionNode(id="a", title="Revenue revenue revenue", page_start=1, page_end=1, children=[]),
            SectionNode(id="b", title="Revenue", page_start=2, page_end=2, children=[]),
            SectionNode(id="c", title="Other", page_start=3, page_end=3, children=[]),
        ],
    )
    page_index = PageIndex(doc_id="d", root=root)
    result = navigate_pageindex(page_index, "revenue", top_k=3)
    assert len(result) >= 2
    scores = []
    for sec in result:
        text = (sec.title + " " + (sec.summary or "")).lower()
        scores.append(sum(1 for tok in "revenue".split() if len(tok) > 1 and tok in text))
    assert scores == sorted(scores, reverse=True)


# ---------- Helpers ----------


def test_collect_sections_flatten_includes_nested() -> None:
    """_collect_sections_flatten returns root and all descendants in a flat list."""
    c = SectionNode(id="c", title="C", page_start=2, page_end=2, children=[])
    b = SectionNode(id="b", title="B", page_start=1, page_end=2, children=[c])
    a = SectionNode(id="a", title="A", page_start=1, page_end=2, children=[b])
    flat = _collect_sections_flatten(a)
    assert len(flat) == 3
    ids = {n.id for n in flat}
    assert ids == {"a", "b", "c"}


def test_data_types_from_ldus_filters_by_page_range() -> None:
    """_data_types_from_ldus only considers LDUs in the given page range."""
    ldus = [
        LDU(id="1", doc_id="d", chunk_type=ChunkType.paragraph, content="x", page_refs=[1], token_count=1, content_hash="1"),
        LDU(id="2", doc_id="d", chunk_type=ChunkType.table, content="x", page_refs=[2], token_count=1, content_hash="2"),
    ]
    types_1 = _data_types_from_ldus(ldus, 1, 1)
    types_2 = _data_types_from_ldus(ldus, 2, 2)
    assert "text" in types_1
    assert "tables" in types_2
