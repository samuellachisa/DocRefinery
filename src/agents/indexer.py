from __future__ import annotations

from typing import List

from src.llm import LLMConfig, call_text_llm
from src.models import ExtractedDocument, PageIndex, SectionNode
from pathlib import Path
import json


class PageIndexBuilder:
    """
    Builds a hierarchical PageIndex.

    This initial implementation builds a simple one-level tree with the whole
    document as a single section and uses an LLM to summarize it.
    """

    def __init__(
        self,
    ) -> None:
        self.llm_cfg = LLMConfig.from_env()

    def _summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following document content in 2-3 sentences, "
            "focusing on key entities, sections, and data types:\n\n"
        )

        try:
            return call_text_llm(prompt + text[:8000], self.llm_cfg)
        except Exception:
            # Offline/dev mode: fall back to trivial message.
            return "Summary not available (LLM backend not configured or unreachable)."

    def build(self, doc: ExtractedDocument) -> PageIndex:
        if not doc.text_blocks:
            summary_text = ""
        else:
            blocks = [tb.text for tb in doc.text_blocks[:20]]
            summary_text = "\n\n".join(blocks)

        # Root summary
        summary = self._summarize(summary_text) if summary_text else ""

        page_numbers = [tb.page_number for tb in doc.text_blocks] or [1]
        page_start = min(page_numbers)
        page_end = max(page_numbers)

        # Simple hierarchy: root + one child per page
        children: List[SectionNode] = []
        for p in range(page_start, page_end + 1):
            page_text = "\n\n".join(
                tb.text for tb in doc.text_blocks if tb.page_number == p
            )
            child_summary = self._summarize(page_text) if page_text else ""
            children.append(
                SectionNode(
                    id=f"{doc.doc_id}-p{p}",
                    title=f"Page {p}",
                    page_start=p,
                    page_end=p,
                    summary=child_summary,
                    key_entities=[],
                    data_types_present=["text", "tables" if doc.tables else "text"],
                    children=[],
                )
            )

        root = SectionNode(
            id=f"{doc.doc_id}-root",
            title="Document",
            page_start=page_start,
            page_end=page_end,
            summary=summary,
            key_entities=[],  # can be populated with NER later
            data_types_present=["text", "tables" if doc.tables else "text"],
            children=children,
        )

        page_index = PageIndex(doc_id=doc.doc_id, root=root)

        # Persist PageIndex JSON artifact
        out_dir = Path(".refinery/pageindex")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{doc.doc_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(page_index.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        return page_index


def navigate_pageindex(page_index: PageIndex, topic: str, top_k: int = 3) -> List[SectionNode]:
    """
    PageIndex query:
    - Scores sections (root children) by simple keyword overlap with topic.
    - Returns top_k sections.
    """
    topic_lower = topic.lower()
    candidates = page_index.root.children or [page_index.root]
    scored = []
    for sec in candidates:
        text = f"{sec.title}\n{sec.summary or ''}".lower()
        score = sum(1 for tok in topic_lower.split() if tok in text)
        scored.append((score, sec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for score, sec in scored[:top_k] if score > 0]

