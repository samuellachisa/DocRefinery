"""Unit tests for Pydantic models."""

from __future__ import annotations

import pytest

from src.models import BoundingBox, DocumentProfile, LDU
from src.models import ChunkType, DomainHint, EstimatedExtractionCost
from src.models import LanguageProfile, LayoutComplexity, OriginType
from src.models.extracted_document import TextBlock


def test_bounding_box_valid() -> None:
    """BoundingBox accepts valid coordinates."""
    bbox = BoundingBox(x0=0.0, y0=0.0, x1=100.0, y1=50.0)
    assert bbox.x1 >= bbox.x0
    assert bbox.y1 >= bbox.y0


def test_bounding_box_rejects_invalid_x1() -> None:
    """BoundingBox rejects x1 < x0."""
    with pytest.raises(ValueError, match="x1 must be greater than or equal to x0"):
        BoundingBox(x0=100.0, y0=0.0, x1=50.0, y1=50.0)


def test_bounding_box_rejects_invalid_y1() -> None:
    """BoundingBox rejects y1 < y0."""
    with pytest.raises(ValueError, match="y1 must be greater than or equal to y0"):
        BoundingBox(x0=0.0, y0=50.0, x1=100.0, y1=10.0)


def test_ldu_requires_non_empty_page_refs() -> None:
    """LDU requires at least one page_ref."""
    with pytest.raises(ValueError, match="page_refs must contain at least one"):
        LDU(
            id="x",
            doc_id="d",
            chunk_type=ChunkType.paragraph,
            content="x",
            page_refs=[],
            token_count=1,
            content_hash="x",
        )


def test_ldu_rejects_zero_page_number() -> None:
    """LDU rejects page numbers < 1."""
    with pytest.raises(ValueError, match="positive page numbers"):
        LDU(
            id="x",
            doc_id="d",
            chunk_type=ChunkType.paragraph,
            content="x",
            page_refs=[0],
            token_count=1,
            content_hash="x",
        )


def test_document_profile_extraction_strategy_mapping() -> None:
    """DocumentProfile.extraction_strategy_needed maps EstimatedExtractionCost correctly."""
    from pathlib import Path

    def _profile(cost: EstimatedExtractionCost) -> DocumentProfile:
        return DocumentProfile(
            doc_id="x",
            source_path=Path("/tmp/x.pdf"),
            num_pages=1,
            origin_type=OriginType.native_digital,
            layout_complexity=LayoutComplexity.single_column,
            language=LanguageProfile(code="en", confidence=1.0),
            domain_hint=DomainHint.general,
            estimated_extraction_cost=cost,
            avg_char_density=0.01,
            avg_image_area_ratio=0.0,
            has_ocr_layer=True,
            table_count_estimate=0,
            triage_confidence=0.9,
        )

    assert _profile(EstimatedExtractionCost.fast_text_sufficient).extraction_strategy_needed == "fast_text"
    assert _profile(EstimatedExtractionCost.needs_layout_model).extraction_strategy_needed == "layout_aware"
    assert _profile(EstimatedExtractionCost.needs_vision_model).extraction_strategy_needed == "vision"
