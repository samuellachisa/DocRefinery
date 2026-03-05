"""Unit tests for KeywordDomainHintClassifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.triage import KeywordDomainHintClassifier, load_triage_config
from src.models import DomainHint


def test_classify_financial() -> None:
    """Classifier returns financial for balance sheet / revenue keywords."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("This balance sheet shows revenue and profit.") == DomainHint.financial
    assert clf.classify("Assets and liabilities for the fiscal year.") == DomainHint.financial


def test_classify_legal() -> None:
    """Classifier returns legal for court / decree keywords."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("The court hereby orders and decrees.") == DomainHint.legal
    assert clf.classify("Pursuant to the plaintiff's motion.") == DomainHint.legal


def test_classify_technical() -> None:
    """Classifier returns technical for architecture / latency keywords."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("This architecture improves throughput and reduces latency.") == DomainHint.technical
    assert clf.classify("Protocol specification and implementation details.") == DomainHint.technical


def test_classify_medical() -> None:
    """Classifier returns medical for clinical / patient keywords."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("Clinical diagnosis and treatment for the patient.") == DomainHint.medical
    assert clf.classify("Disease symptoms and therapy options.") == DomainHint.medical


def test_classify_general_when_no_match() -> None:
    """Classifier returns general when no domain keywords match."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("The weather is nice today.") == DomainHint.general
    assert clf.classify("Hello world.") == DomainHint.general


def test_classify_case_insensitive() -> None:
    """Classifier matches keywords case-insensitively."""
    clf = KeywordDomainHintClassifier()
    assert clf.classify("BALANCE SHEET and REVENUE") == DomainHint.financial
    assert clf.classify("COURT and DECREE") == DomainHint.legal


def test_from_config_uses_rules_keywords(rules_path: Path) -> None:
    """Classifier built from config uses domain_keywords from extraction_rules."""
    cfg = load_triage_config(rules_path)
    clf = KeywordDomainHintClassifier.from_config(cfg)
    assert clf.classify("balance sheet and income statement") == DomainHint.financial
    assert clf.classify("hereinafter and pursuant to") == DomainHint.legal
