from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.agents.triage import profile_document
from src.models import DomainHint, EstimatedExtractionCost, LayoutComplexity, OriginType


def _make_simple_pdf(tmp_path: Path, text: str) -> Path:
    pdf_path = tmp_path / "simple.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, text)
    c.showPage()
    c.save()
    return pdf_path


@pytest.mark.parametrize(
    "text, expected_domain",
    [
        ("This balance sheet shows revenue and profit for the fiscal year.", DomainHint.financial),
        ("The court hereby orders and decrees as follows.", DomainHint.legal),
        ("This architecture improves throughput and reduces latency.", DomainHint.technical),
    ],
)
def test_domain_hint_classification(tmp_path: Path, text: str, expected_domain: DomainHint) -> None:
    pdf_path = _make_simple_pdf(tmp_path, text)
    profile = profile_document(pdf_path)
    assert profile.domain_hint == expected_domain


def test_origin_type_native_digital_single_column(tmp_path: Path) -> None:
    text = "This is a simple native digital PDF document for testing."
    pdf_path = _make_simple_pdf(tmp_path, text)
    profile = profile_document(pdf_path)

    assert profile.origin_type in {OriginType.native_digital, OriginType.mixed}
    assert profile.layout_complexity == LayoutComplexity.single_column
    assert profile.estimated_extraction_cost in {
        EstimatedExtractionCost.fast_text_sufficient,
        EstimatedExtractionCost.needs_layout_model,
    }

