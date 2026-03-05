"""Unit tests for extraction_rules.yaml config loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.chunker import load_chunking_config
from src.agents.extractor import load_router_config
from src.agents.triage import load_triage_config


def test_load_triage_config(rules_path: Path) -> None:
    """Triage config loads with expected structure."""
    cfg = load_triage_config(rules_path)
    assert cfg.native_digital_min_char_density == 0.002
    assert cfg.scanned_max_char_density == 0.0003
    assert cfg.scanned_min_image_ratio == 0.4
    assert cfg.fast_text_max_image_ratio == 0.3
    assert cfg.multi_column_min_xbands == 2
    assert cfg.table_heavy_table_ratio == 0.2
    assert cfg.figure_heavy_figure_ratio == 0.2
    assert cfg.base_confidence == 0.7
    assert "financial" in cfg.domain_keywords
    assert "balance sheet" in cfg.domain_keywords["financial"]


def test_load_router_config(rules_path: Path) -> None:
    """Router config loads with escalation thresholds."""
    cfg = load_router_config(rules_path)
    assert cfg.fast_text_min_page_coverage == 0.9
    assert cfg.fast_text_min_avg_confidence == 0.6
    assert cfg.layout_min_avg_confidence == 0.6


def test_load_chunking_config(rules_path: Path) -> None:
    """Chunking config loads with token budgets."""
    cfg = load_chunking_config(rules_path)
    assert cfg.max_tokens_per_chunk == 512
    assert cfg.hard_max_tokens_per_chunk == 1024
    assert cfg.keep_numbered_lists_together is True


def test_load_router_config_with_missing_keys(tmp_path: Path) -> None:
    """Router config uses defaults when keys are missing."""
    minimal_yaml = tmp_path / "minimal.yaml"
    minimal_yaml.write_text("extraction:\n  fast_text: {}\n  layout: {}\n")
    cfg = load_router_config(minimal_yaml)
    assert cfg.fast_text_min_page_coverage == 0.9
    assert cfg.fast_text_min_avg_confidence == 0.6
    assert cfg.layout_min_avg_confidence == 0.6


def test_load_chunking_config_with_missing_keys(tmp_path: Path) -> None:
    """Chunking config uses defaults when keys are missing."""
    minimal_yaml = tmp_path / "minimal.yaml"
    minimal_yaml.write_text("chunking: {}\n")
    cfg = load_chunking_config(minimal_yaml)
    assert cfg.max_tokens_per_chunk == 512
    assert cfg.hard_max_tokens_per_chunk == 1024
    assert cfg.keep_numbered_lists_together is True
