"""Shared pytest fixtures for DocRefinery tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def make_simple_pdf(tmp_path: Path, text: str, filename: str = "simple.pdf") -> Path:
    """Create a minimal PDF with the given text."""
    pdf_path = tmp_path / filename
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica", 12)
    c.drawString(72, height - 72, text)
    c.showPage()
    c.save()
    return pdf_path


@pytest.fixture
def rules_path() -> Path:
    """Path to extraction_rules.yaml."""
    return Path("rubric/extraction_rules.yaml")
