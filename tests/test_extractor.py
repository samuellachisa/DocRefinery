"""Unit tests for ExtractionRouter and config loading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.extractor import (
    ExtractionRouter,
    load_router_config,
)
from src.models import DocumentProfile, EstimatedExtractionCost, ExtractedDocument
from src.models import LanguageProfile, OriginType, LayoutComplexity, DomainHint
from src.strategies import ExtractionResult


def _make_profile(
    estimated_cost: EstimatedExtractionCost = EstimatedExtractionCost.fast_text_sufficient,
    num_pages: int = 5,
) -> DocumentProfile:
    """Build a minimal DocumentProfile for testing."""
    return DocumentProfile(
        doc_id="test-doc-123",
        source_path=Path("/tmp/test.pdf"),
        num_pages=num_pages,
        origin_type=OriginType.native_digital,
        layout_complexity=LayoutComplexity.single_column,
        language=LanguageProfile(code="en", confidence=0.95),
        domain_hint=DomainHint.financial,
        estimated_extraction_cost=estimated_cost,
        avg_char_density=0.003,
        avg_image_area_ratio=0.1,
        has_ocr_layer=True,
        table_count_estimate=0,
        triage_confidence=0.85,
    )


def test_load_router_config(rules_path: Path) -> None:
    """Router config loads with expected values."""
    cfg = load_router_config(rules_path)
    assert cfg.fast_text_min_page_coverage == 0.9
    assert cfg.fast_text_min_avg_confidence == 0.6
    assert cfg.layout_min_avg_confidence == 0.6


def test_router_escalates_fast_text_when_coverage_low(rules_path: Path) -> None:
    """Router escalates from fast_text to layout when page coverage is below threshold."""
    profile = _make_profile(num_pages=10)
    # Mock fast_text to return low coverage (only 2 good pages out of 10 = 0.2 < 0.9)
    low_coverage_confidences = {i: 0.1 for i in range(1, 9)}  # pages 1-8: low
    low_coverage_confidences[9] = 0.8  # page 9: good
    low_coverage_confidences[10] = 0.8  # page 10: good
    # good_pages = those with c >= 0.3, so only 9 and 10 -> coverage = 0.2
    mock_fast_result = ExtractionResult(
        document=ExtractedDocument(doc_id=profile.doc_id, num_pages=10),
        page_confidences=low_coverage_confidences,
        strategy_name="fast_text",
        cost_estimate_usd=0.0,
    )

    with patch.object(ExtractionRouter, "_log_ledger"):
        router = ExtractionRouter(rules_path)
        with patch.object(router.fast_text, "extract", return_value=mock_fast_result):
            with patch.object(router.layout, "extract") as mock_layout:
                mock_layout.return_value = ExtractionResult(
                    document=ExtractedDocument(doc_id=profile.doc_id, num_pages=10),
                    page_confidences={i: 0.9 for i in range(1, 11)},
                    strategy_name="layout",
                    cost_estimate_usd=0.01,
                )
                result = router.route(profile)
                assert result.strategy_name == "layout"
                mock_layout.assert_called_once_with(profile)


def test_router_does_not_escalate_when_fast_text_sufficient(rules_path: Path) -> None:
    """Router stays on fast_text when coverage and confidence are adequate."""
    profile = _make_profile(num_pages=5)
    high_confidences = {i: 0.9 for i in range(1, 6)}
    mock_fast_result = ExtractionResult(
        document=ExtractedDocument(doc_id=profile.doc_id, num_pages=5),
        page_confidences=high_confidences,
        strategy_name="fast_text",
        cost_estimate_usd=0.0,
    )

    with patch.object(ExtractionRouter, "_log_ledger"):
        router = ExtractionRouter(rules_path)
        with patch.object(router.fast_text, "extract", return_value=mock_fast_result):
            result = router.route(profile)
            assert result.strategy_name == "fast_text"


def test_router_selects_layout_for_needs_layout_profile(rules_path: Path) -> None:
    """Router uses layout strategy when profile says needs_layout_model."""
    profile = _make_profile(
        estimated_cost=EstimatedExtractionCost.needs_layout_model
    )
    mock_layout_result = ExtractionResult(
        document=ExtractedDocument(doc_id=profile.doc_id, num_pages=5),
        page_confidences={i: 0.9 for i in range(1, 6)},
        strategy_name="layout",
        cost_estimate_usd=0.01,
    )

    with patch.object(ExtractionRouter, "_log_ledger"):
        router = ExtractionRouter(rules_path)
        with patch.object(router.layout, "extract", return_value=mock_layout_result):
            result = router.route(profile)
            assert result.strategy_name == "layout"
