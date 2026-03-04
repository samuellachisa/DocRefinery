from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import pdfplumber
import yaml
from langdetect import detect, DetectorFactory

from src.models import (
    DocumentProfile,
    DomainHint,
    EstimatedExtractionCost,
    LanguageProfile,
    LayoutComplexity,
    OriginType,
)

DetectorFactory.seed = 0  # make language detection deterministic


@dataclass
class TriageConfig:
    native_digital_min_char_density: float
    scanned_max_char_density: float
    scanned_min_image_ratio: float
    fast_text_max_image_ratio: float
    multi_column_min_xbands: int
    table_heavy_table_ratio: float
    figure_heavy_figure_ratio: float
    base_confidence: float
    bonus_clear_origin: float
    bonus_clear_layout: float
    domain_keywords: Dict[str, List[str]]


def load_triage_config(config_path: Path) -> TriageConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    triage_cfg = data.get("triage", {})
    layout_cfg = triage_cfg.get("layout", {})
    confidence_cfg = triage_cfg.get("confidence", {})
    domain_kw_cfg = triage_cfg.get("domain_keywords", {}) or {}

    return TriageConfig(
        native_digital_min_char_density=triage_cfg["char_density"][
            "native_digital_min"
        ],
        scanned_max_char_density=triage_cfg["char_density"]["scanned_max"],
        scanned_min_image_ratio=triage_cfg["image_area_ratio"]["scanned_min"],
        fast_text_max_image_ratio=triage_cfg["image_area_ratio"]["fast_text_max"],
        multi_column_min_xbands=layout_cfg["multi_column_min_xbands"],
        table_heavy_table_ratio=layout_cfg["table_heavy_table_ratio"],
        figure_heavy_figure_ratio=layout_cfg["figure_heavy_figure_ratio"],
        base_confidence=confidence_cfg["base"],
        bonus_clear_origin=confidence_cfg["bonus_clear_origin"],
        bonus_clear_layout=confidence_cfg["bonus_clear_layout"],
        domain_keywords={
            str(domain): list(keywords or [])
            for domain, keywords in domain_kw_cfg.items()
        },
    )


class DomainHintClassifier:
    """Pluggable classifier for domain hints."""

    def classify(self, text_sample: str) -> DomainHint:
        raise NotImplementedError


class KeywordDomainHintClassifier(DomainHintClassifier):
    """Keyword-based classifier driven by config, easily swappable with a VLM-based one."""

    def __init__(self, keywords_by_domain: Optional[Dict[DomainHint, List[str]]] = None):
        if keywords_by_domain is None:
            # Fallback to a small built-in set; in normal operation we expect
            # keywords to be provided via TriageConfig.
            keywords_by_domain = {
                DomainHint.financial: [
                    "balance sheet",
                    "income statement",
                    "statement of cash flows",
                    "revenue",
                    "profit",
                    "loss",
                    "equity",
                    "assets",
                    "liabilities",
                    "fiscal year",
                    "tax",
                ],
                DomainHint.legal: [
                    "hereinafter",
                    "plaintiff",
                    "defendant",
                    "court",
                    "decree",
                    "pursuant to",
                    "whereas",
                ],
                DomainHint.technical: [
                    "algorithm",
                    "architecture",
                    "throughput",
                    "latency",
                    "protocol",
                    "specification",
                    "implementation",
                ],
                DomainHint.medical: [
                    "clinical",
                    "patient",
                    "diagnosis",
                    "treatment",
                    "therapy",
                    "symptom",
                    "disease",
                ],
            }
        self.keywords_by_domain = {
            domain: [kw.lower() for kw in keywords]
            for domain, keywords in keywords_by_domain.items()
        }

    @classmethod
    def from_config(cls, cfg: TriageConfig) -> "KeywordDomainHintClassifier":
        mapping: Dict[DomainHint, List[str]] = {}
        for key, keywords in cfg.domain_keywords.items():
            try:
                domain = DomainHint(key)
            except ValueError:
                # Ignore unknown domain hints so config can evolve safely.
                continue
            mapping[domain] = keywords
        # If config didn't specify anything, we'll fall back to built-ins.
        if not mapping:
            return cls()
        return cls(mapping)

    def classify(self, text_sample: str) -> DomainHint:
        lower = text_sample.lower()
        for domain, keywords in self.keywords_by_domain.items():
            if any(k in lower for k in keywords):
                return domain
        return DomainHint.general


def _iter_page_stats(pdf: pdfplumber.PDF) -> Iterable[Dict[str, float]]:
    for page in pdf.pages:
        width = float(page.width or 1)
        height = float(page.height or 1)
        area = width * height

        chars = page.chars or []
        char_count = float(len(chars))
        char_density = char_count / area if area > 0 else 0.0

        # whitespace approximation: measure fraction of empty lines vs lines with chars
        text = page.extract_text() or ""
        lines = [ln for ln in (text.splitlines() if text else [])]
        if lines:
            empty_lines = sum(1 for ln in lines if not ln.strip())
            whitespace_ratio = empty_lines / float(len(lines))
        else:
            whitespace_ratio = 1.0

        images = page.images or []
        image_area = 0.0
        for img in images:
            iw = float(img.get("width", 0) or 0)
            ih = float(img.get("height", 0) or 0)
            image_area += iw * ih
        image_area_ratio = image_area / area if area > 0 else 0.0

        # simple table / figure proxies
        try:
            table_candidates = page.extract_tables() or []
        except Exception:
            table_candidates = []
        table_count = float(len(table_candidates))
        figure_count = float(len(images))

        yield {
            "char_count": char_count,
            "char_density": char_density,
            "image_area_ratio": image_area_ratio,
            "whitespace_ratio": whitespace_ratio,
            "table_count": table_count,
            "figure_count": figure_count,
            "width": width,
            "height": height,
        }


def _estimate_layout_complexity(pdf: pdfplumber.PDF, cfg: TriageConfig) -> LayoutComplexity:
    x_positions: List[float] = []
    for page in pdf.pages:
        for char in page.chars or []:
            x_positions.append(float(char.get("x0", 0.0)))

    if not x_positions:
        return LayoutComplexity.single_column

    # Very simple band-based detection: cluster x positions into buckets of fixed width
    min_x, max_x = min(x_positions), max(x_positions)
    span = max(max_x - min_x, 1.0)
    band_width = span / 6.0  # 6 vertical bands across the page
    band_counts: Dict[int, int] = {}

    for x in x_positions:
        band_idx = int((x - min_x) / band_width)
        band_counts[band_idx] = band_counts.get(band_idx, 0) + 1

    # Count how many bands hold significant text
    total_chars = len(x_positions)
    significant_bands = [
        idx for idx, count in band_counts.items() if count / total_chars > 0.15
    ]

    if len(significant_bands) >= cfg.multi_column_min_xbands:
        return LayoutComplexity.multi_column

    # Fallback – more nuanced detection would look at tables/figures, but we don't have them yet
    return LayoutComplexity.single_column


def _detect_language(text_sample: str) -> LanguageProfile:
    try:
        code = detect(text_sample)
        confidence = 0.9  # langdetect does not expose probability; we treat as high
    except Exception:
        code = "unknown"
        confidence = 0.0
    return LanguageProfile(code=code, confidence=confidence)


def _compute_doc_id(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.name.encode("utf-8"))
    h.update(str(path.stat().st_size).encode("utf-8"))
    return h.hexdigest()


def profile_document(
    pdf_path: Path,
    rules_path: Path | None = None,
    domain_classifier: Optional[DomainHintClassifier] = None,
) -> DocumentProfile:
    if rules_path is None:
        rules_path = Path("rubric/extraction_rules.yaml")

    cfg = load_triage_config(rules_path)

    if domain_classifier is None:
        domain_classifier = KeywordDomainHintClassifier.from_config(cfg)

    with pdfplumber.open(pdf_path) as pdf:
        page_stats = list(_iter_page_stats(pdf))

        num_pages = len(pdf.pages)
        avg_char_density = (
            mean(p["char_density"] for p in page_stats) if page_stats else 0.0
        )
        avg_image_area_ratio = (
            mean(p["image_area_ratio"] for p in page_stats) if page_stats else 0.0
        )
        avg_whitespace_ratio = (
            mean(p["whitespace_ratio"] for p in page_stats) if page_stats else 0.0
        )
        avg_table_count = (
            mean(p["table_count"] for p in page_stats) if page_stats else 0.0
        )
        avg_figure_count = (
            mean(p["figure_count"] for p in page_stats) if page_stats else 0.0
        )

        table_pages_with_content = sum(
            1 for p in page_stats if p["table_count"] > 0.0
        )
        figure_pages_with_content = sum(
            1 for p in page_stats if p["figure_count"] > 0.0
        )
        table_pages_ratio = (
            table_pages_with_content / float(num_pages or 1)
        )
        figure_pages_ratio = (
            figure_pages_with_content / float(num_pages or 1)
        )

        # Collect sample text from first few pages
        sample_text_parts: List[str] = []
        for page in pdf.pages[: min(5, num_pages)]:
            txt = page.extract_text() or ""
            if txt:
                sample_text_parts.append(txt)
        sample_text = "\n".join(sample_text_parts)[:8000]

    # Origin type heuristic
    if (
        avg_char_density >= cfg.native_digital_min_char_density
        and avg_image_area_ratio <= cfg.fast_text_max_image_ratio
    ):
        origin_type = OriginType.native_digital
        clear_origin = True
    elif (
        avg_char_density <= cfg.scanned_max_char_density
        and avg_image_area_ratio >= cfg.scanned_min_image_ratio
    ):
        origin_type = OriginType.scanned_image
        clear_origin = True
    else:
        origin_type = OriginType.mixed
        clear_origin = False

    # Layout complexity heuristic (requires pdf again)
    with pdfplumber.open(pdf_path) as pdf2:
        layout_complexity = _estimate_layout_complexity(pdf2, cfg)

    # refine layout complexity using table / figure density ratios from config
    if (
        table_pages_ratio >= cfg.table_heavy_table_ratio
        and layout_complexity is not LayoutComplexity.multi_column
    ):
        layout_complexity = LayoutComplexity.table_heavy
    elif (
        figure_pages_ratio >= cfg.figure_heavy_figure_ratio
        and layout_complexity is not LayoutComplexity.multi_column
    ):
        layout_complexity = LayoutComplexity.figure_heavy

    clear_layout = layout_complexity is not LayoutComplexity.mixed

    # Language and domain
    language = _detect_language(sample_text)
    domain_hint = domain_classifier.classify(sample_text)

    # Estimated extraction cost
    # Special-case optimisation: simple native-digital financial docs (e.g. single-page invoices)
    # can safely use fast text extraction even when a small table is present.
    simple_financial_doc = (
        origin_type is OriginType.native_digital
        and layout_complexity in {LayoutComplexity.single_column, LayoutComplexity.table_heavy}
        and domain_hint is DomainHint.financial
        and num_pages <= 2
        and avg_table_count <= 2.0
        and avg_image_area_ratio <= cfg.fast_text_max_image_ratio
    )

    if simple_financial_doc:
        estimated_cost = EstimatedExtractionCost.fast_text_sufficient
    elif origin_type is OriginType.native_digital and layout_complexity is LayoutComplexity.single_column:
        estimated_cost = EstimatedExtractionCost.fast_text_sufficient
    elif origin_type is OriginType.scanned_image:
        estimated_cost = EstimatedExtractionCost.needs_vision_model
    else:
        estimated_cost = EstimatedExtractionCost.needs_layout_model

    triage_confidence = cfg.base_confidence
    if clear_origin:
        triage_confidence += cfg.bonus_clear_origin
    if clear_layout:
        triage_confidence += cfg.bonus_clear_layout
    # whitespace extremes lower confidence (either too sparse or too dense)
    if avg_whitespace_ratio < 0.05 or avg_whitespace_ratio > 0.9:
        triage_confidence -= 0.1
    triage_confidence = max(0.0, min(1.0, triage_confidence))

    # Empirical signals for OCR and table estimates
    # Heuristic: if there is non-trivial text density, treat as having an OCR/text layer.
    has_ocr_layer = avg_char_density > cfg.scanned_max_char_density
    table_count_estimate = int(round(avg_table_count * num_pages))

    doc_id = _compute_doc_id(pdf_path)

    profile = DocumentProfile(
        doc_id=doc_id,
        source_path=pdf_path,
        num_pages=num_pages,
        origin_type=origin_type,
        layout_complexity=layout_complexity,
        language=language,
        domain_hint=domain_hint,
        estimated_extraction_cost=estimated_cost,
        avg_char_density=avg_char_density,
        avg_image_area_ratio=avg_image_area_ratio,
        has_ocr_layer=has_ocr_layer,
        table_count_estimate=table_count_estimate,
        triage_confidence=triage_confidence,
        notes=f"avg_whitespace_ratio={avg_whitespace_ratio:.3f}, "
        f"avg_table_count={avg_table_count:.2f}, avg_figure_count={avg_figure_count:.2f}",
    )

    # Persist profile for downstream stages
    profiles_dir = Path(".refinery/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out_path = profiles_dir / f"{doc_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(profile.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

    return profile


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Triage Agent – Document Profiling")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF document.")
    parser.add_argument(
        "--rules",
        type=str,
        default="rubric/extraction_rules.yaml",
        help="Path to extraction_rules.yaml.",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    rules_path = Path(args.rules).expanduser().resolve()

    profile = profile_document(pdf_path, rules_path)
    print(json.dumps(profile.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

