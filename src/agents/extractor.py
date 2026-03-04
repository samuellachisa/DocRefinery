from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

import yaml

from src.models import DocumentProfile, EstimatedExtractionCost
from src.strategies import (
    ExtractionResult,
    FastTextExtractor,
    LayoutExtractor,
    VisionExtractor,
)


@dataclass
class RouterConfig:
    fast_text_min_page_coverage: float
    fast_text_min_avg_confidence: float
    layout_min_avg_confidence: float


def load_router_config(config_path: Path) -> RouterConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    extraction_cfg = data.get("extraction", {})
    fast_text_cfg = extraction_cfg.get("fast_text", {})
    layout_cfg = extraction_cfg.get("layout", {})
    return RouterConfig(
        fast_text_min_page_coverage=float(fast_text_cfg.get("min_page_coverage", 0.9)),
        fast_text_min_avg_confidence=float(
            fast_text_cfg.get("min_avg_confidence", 0.6)
        ),
        layout_min_avg_confidence=float(layout_cfg.get("min_avg_confidence", 0.6)),
    )


class ExtractionRouter:
    """Strategy pattern router with escalation guard and extraction ledger logging."""

    def __init__(self, rules_path: Path | None = None) -> None:
        if rules_path is None:
            rules_path = Path("rubric/extraction_rules.yaml")
        self.rules_path = rules_path
        self.cfg = load_router_config(rules_path)
        self.fast_text = FastTextExtractor()
        self.layout = LayoutExtractor()
        self.vision = VisionExtractor(rules_path=rules_path)

    def _log_ledger(
        self,
        profile: DocumentProfile,
        result: ExtractionResult,
        strategy_used: str,
        escalated_from: Optional[str],
        processing_time_s: float,
    ) -> None:
        ledger_path = Path(".refinery/extraction_ledger.jsonl")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

        avg_conf = (
            mean(result.page_confidences.values())
            if result.page_confidences
            else 0.0
        )
        entry = {
            "doc_id": profile.doc_id,
            "source_path": str(profile.source_path),
            "strategy_used": strategy_used,
            "escalated_from": escalated_from,
            "avg_page_confidence": avg_conf,
            "cost_estimate_usd": result.cost_estimate_usd,
            "processing_time_s": processing_time_s,
        }
        with ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _needs_escalation_fast_text(
        self, profile: DocumentProfile, result: ExtractionResult
    ) -> bool:
        if not result.page_confidences:
            return True
        avg_conf = mean(result.page_confidences.values())
        # page coverage: proportion of pages with non-trivial confidence
        good_pages = [
            p for p, c in result.page_confidences.items() if c >= 0.3
        ]
        coverage = len(good_pages) / float(profile.num_pages or 1)
        return bool(
            coverage < self.cfg.fast_text_min_page_coverage
            or avg_conf < self.cfg.fast_text_min_avg_confidence
        )

    def _needs_escalation_layout(self, result: ExtractionResult) -> bool:
        if not result.page_confidences:
            return True
        avg_conf = mean(result.page_confidences.values())
        return avg_conf < self.cfg.layout_min_avg_confidence

    def route(self, profile: DocumentProfile) -> ExtractionResult:
        """Run appropriate extraction strategy with escalation guard."""
        start = time.time()
        escalated_from: Optional[str] = None

        if profile.estimated_extraction_cost is EstimatedExtractionCost.fast_text_sufficient:
            primary = self.fast_text
        elif profile.estimated_extraction_cost is EstimatedExtractionCost.needs_layout_model:
            primary = self.layout
        else:
            primary = self.vision

        result = primary.extract(profile)
        strategy_used = primary.name

        # Escalation guard
        if primary is self.fast_text and self._needs_escalation_fast_text(
            profile, result
        ):
            escalated_from = strategy_used
            secondary = self.layout
            result = secondary.extract(profile)
            strategy_used = secondary.name

        if (
            strategy_used == self.layout.name
            and self._needs_escalation_layout(result)
        ):
            if escalated_from is None:
                escalated_from = strategy_used
            result = self.vision.extract(profile)
            strategy_used = self.vision.name

        elapsed = time.time() - start
        self._log_ledger(profile, result, strategy_used, escalated_from, elapsed)
        return result


def main() -> None:
    import argparse

    from src.agents.triage import profile_document

    parser = argparse.ArgumentParser(
        description="Extraction Router – multi-strategy extraction with escalation guard"
    )
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

    router = ExtractionRouter(rules_path)
    result = router.route(profile)

    # Persist the normalized ExtractedDocument so downstream steps (like chunking)
    # can be run later without re-doing extraction.
    extracted_dir = Path(".refinery/extracted")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = extracted_dir / f"{profile.doc_id}.json"
    with extracted_path.open("w", encoding="utf-8") as f:
        json.dump(
            result.document.model_dump(mode="json"),
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Print a brief summary for CLI usage
    print(
        json.dumps(
            {
                "doc_id": profile.doc_id,
                "strategy_used": result.strategy_name,
                "num_text_blocks": len(result.document.text_blocks),
                "num_tables": len(result.document.tables),
                "num_figures": len(result.document.figures),
                "extracted_path": str(extracted_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

