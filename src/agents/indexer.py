from __future__ import annotations

import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import json

from src.llm import LLMConfig, call_text_llm
from src.models import ExtractedDocument, LDU, PageIndex, SectionNode
from src.models.ldu import ChunkType


def _detect_headings(text_blocks: List) -> List[Tuple[int, str, int]]:
    """
    Detect section headings from text blocks (reading order).
    Returns list of (page_number, title, level).
    Level 1: Chapter N, Part N, Section N; 2: N.M or N. Title; 3: N.M.K.
    """
    headings: List[Tuple[int, str, int]] = []
    chapter_re = re.compile(r"^(chapter|part|section)\s+\d+[\s.:]*(.*)$", re.IGNORECASE)
    numbered_re = re.compile(r"^(\d+)(\.\d+)*\.?\s+([A-Za-z].*)$")
    for tb in text_blocks:
        text = (getattr(tb, "text", None) or "").strip()
        if not text:
            continue
        if not text:
            continue
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            if len(line) > 120:
                continue
            # Chapter / Part / Section
            m = chapter_re.match(line)
            if m:
                title = (m.group(2) or line).strip() or line[:50]
                headings.append((tb.page_number, title, 1))
                continue
            # Numbered: 1. 1.1 2.3.1
            m = numbered_re.match(line)
            if m:
                dots = (m.group(2) or "") or ""
                depth = min(1 + dots.count("."), 3)
                title = (m.group(3) or line).strip()[:80]
                if len(title) >= 2:
                    headings.append((tb.page_number, title, min(depth, 3)))
                continue
            # Short all-caps line as heading
            if line.isupper() and 3 <= len(line) <= 80:
                headings.append((tb.page_number, line, 2))
    return headings


def _build_section_tree(
    doc_id: str,
    doc: ExtractedDocument,
    headings: List[Tuple[int, str, int]],
    llm_cfg: LLMConfig,
) -> SectionNode:
    """Build hierarchical section tree from detected headings and fill summaries + key_entities."""
    page_numbers = [getattr(tb, "page_number", 1) for tb in doc.text_blocks] or [1]
    page_start = min(page_numbers)
    page_end = max(page_numbers)

    def _section_text(page_lo: int, page_hi: int) -> str:
        parts = []
        for tb in doc.text_blocks:
            pn = getattr(tb, "page_number", 0)
            txt = getattr(tb, "text", None) or ""
            if page_lo <= pn <= page_hi and txt:
                parts.append(txt.strip())
        return "\n\n".join(parts[:30])[:8000]

    def _summarize(text: str) -> str:
        if not text.strip():
            return ""
        prompt = (
            "Summarize the following document content in 2-3 sentences, "
            "focusing on key entities, sections, and data types:\n\n"
        )
        try:
            return call_text_llm(prompt + text, llm_cfg)
        except Exception:
            return ""

    def _extract_entities(summary: str, section_text: str) -> List[str]:
        """LLM-extract 3-5 key entities from summary/section."""
        if not summary and not section_text.strip():
            return []
        prompt = (
            "From the following text, list 3 to 5 key entities (names, terms, concepts) "
            "one per line, no numbering. Only the entities:\n\n"
        )
        src = (summary or "") + "\n\n" + (section_text[:2000] or "")
        try:
            raw = call_text_llm(prompt + src, llm_cfg)
            return [ln.strip() for ln in raw.splitlines() if ln.strip()][:5]
        except Exception:
            return []

    # Build flat section nodes with page ranges
    nodes: List[Tuple[int, str, int, int, int]] = []  # level, title, page_start, page_end, idx
    for i, (page, title, level) in enumerate(headings):
        next_page = headings[i + 1][0] - 1 if i + 1 < len(headings) else page_end
        page_end_section = min(next_page, page_end) if next_page >= page else page
        nodes.append((level, title, page, page_end_section, i))

    # Build tree: parent of i is last j < i with level < level(i)
    def make_node(
        level: int, title: str, ps: int, pe: int, idx: int
    ) -> SectionNode:
        section_text = _section_text(ps, pe)
        summary = _summarize(section_text)
        key_entities = _extract_entities(summary, section_text)
        return SectionNode(
            id=f"{doc_id}-sec-{idx}",
            title=title,
            page_start=ps,
            page_end=pe,
            summary=summary or None,
            key_entities=key_entities,
            data_types_present=[],  # filled by caller from LDUs
            children=[],
        )

    section_nodes: List[SectionNode] = [
        make_node(lev, tit, ps, pe, idx) for lev, tit, ps, pe, idx in nodes
    ]
    # Attach children to parents by level
    root = SectionNode(
        id=f"{doc_id}-root",
        title="Document",
        page_start=page_start,
        page_end=page_end,
        summary="",
        key_entities=[],
        data_types_present=[],
        children=[],
    )
    stack: List[Tuple[int, SectionNode]] = [(0, root)]
    for i, sn in enumerate(section_nodes):
        lev, _, _, _, _ = nodes[i]
        while len(stack) > 1 and stack[-1][0] >= lev:
            stack.pop()
        parent = stack[-1][1]
        parent.children.append(sn)
        stack.append((lev, sn))

    # Root summary and entities from full doc
    full_text = _section_text(page_start, page_end)
    root.summary = _summarize(full_text)
    root.key_entities = _extract_entities(root.summary or "", full_text)
    return root


def _data_types_from_ldus(ldus: List[LDU], page_start: int, page_end: int) -> List[Literal["text", "tables", "figures", "equations"]]:
    """Infer data_types_present from LDUs in the given page range."""
    types = set[str]()
    for ldu in ldus:
        if not ldu.page_refs or ldu.page_refs[0] < page_start or ldu.page_refs[0] > page_end:
            continue
        if ldu.chunk_type in (ChunkType.paragraph, ChunkType.list, ChunkType.header, ChunkType.section_header, ChunkType.caption, ChunkType.footnote, ChunkType.table_cell_group):
            types.add("text")
        elif ldu.chunk_type in (ChunkType.table,):
            types.add("tables")
        elif ldu.chunk_type == ChunkType.figure:
            types.add("figures")
    return list(types) if types else ["text"]


class PageIndexBuilder:
    """
    Builds a hierarchical PageIndex with section detection, LLM summaries,
    key entities, and data types derived from LDUs.
    """

    def __init__(self) -> None:
        self.llm_cfg = LLMConfig.from_env()

    def _summarize(self, text: str) -> str:
        prompt = (
            "Summarize the following document content in 2-3 sentences, "
            "focusing on key entities, sections, and data types:\n\n"
        )
        try:
            return call_text_llm(prompt + text[:8000], self.llm_cfg)
        except Exception:
            return "Summary not available (LLM backend not configured or unreachable)."

    def build(self, doc: ExtractedDocument, ldus: Optional[List[LDU]] = None) -> PageIndex:
        if not doc.text_blocks:
            page_start = 1
            page_end = doc.num_pages or 1
            root = SectionNode(
                id=f"{doc.doc_id}-root",
                title="Document",
                page_start=page_start,
                page_end=page_end,
                summary="",
                key_entities=[],
                data_types_present=["text"],
                children=[],
            )
            return PageIndex(doc_id=doc.doc_id, root=root)

        headings = _detect_headings(doc.text_blocks)
        if headings:
            root = _build_section_tree(doc.doc_id, doc, headings, self.llm_cfg)
            # Fill data_types_present from LDUs
            def fill_data_types(node: SectionNode) -> None:
                node.data_types_present = _data_types_from_ldus(
                    ldus or [], node.page_start, node.page_end
                )
                for ch in node.children:
                    fill_data_types(ch)
            fill_data_types(root)
            root.data_types_present = _data_types_from_ldus(
                ldus or [], root.page_start, root.page_end
            )
        else:
            # Fallback: flat page-level structure
            page_start = min(tb.page_number for tb in doc.text_blocks)
            page_end = max(tb.page_number for tb in doc.text_blocks)
            blocks = [tb.text for tb in doc.text_blocks[:20] if hasattr(tb, "text")]
            summary_text = "\n\n".join(blocks) if blocks else ""
            summary = self._summarize(summary_text) if summary_text else ""
            children: List[SectionNode] = []
            for p in range(page_start, page_end + 1):
                page_text = "\n\n".join(
                    getattr(tb, "text", "") or ""
                    for tb in doc.text_blocks
                    if getattr(tb, "page_number", 0) == p
                )
                child_summary = self._summarize(page_text) if page_text else ""
                section_text = page_text
                key_entities: List[str] = []
                try:
                    prompt_ent = "List 3-5 key entities from this text, one per line:\n\n" + (section_text[:2000] or "")
                    raw = call_text_llm(prompt_ent, self.llm_cfg)
                    key_entities = [ln.strip() for ln in raw.splitlines() if ln.strip()][:5]
                except Exception:
                    pass
                data_types = _data_types_from_ldus(ldus or [], p, p)
                children.append(
                    SectionNode(
                        id=f"{doc.doc_id}-p{p}",
                        title=f"Page {p}",
                        page_start=p,
                        page_end=p,
                        summary=child_summary,
                        key_entities=key_entities,
                        data_types_present=data_types,
                        children=[],
                    )
                )
            root = SectionNode(
                id=f"{doc.doc_id}-root",
                title="Document",
                page_start=page_start,
                page_end=page_end,
                summary=summary,
                key_entities=[],
                data_types_present=_data_types_from_ldus(ldus or [], page_start, page_end),
                children=children,
            )

        page_index = PageIndex(doc_id=doc.doc_id, root=root)
        out_dir = Path(".refinery/pageindex")
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / f"{doc.doc_id}.json").open("w", encoding="utf-8") as f:
            json.dump(page_index.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
        return page_index


def _section_for_page(node: SectionNode, page: int) -> Optional[SectionNode]:
    """Return the deepest section node that covers the given page."""
    if not (node.page_start <= page <= node.page_end):
        return None
    for child in node.children:
        found = _section_for_page(child, page)
        if found is not None:
            return found
    return node


def enrich_ldus_with_sections(ldus: List[LDU], page_index: PageIndex) -> None:
    """Set each LDU's parent_section to the section title covering its first page."""
    for ldu in ldus:
        if not ldu.page_refs:
            continue
        page = ldu.page_refs[0]
        section = _section_for_page(page_index.root, page)
        if section is not None:
            ldu.parent_section = section.title


def _collect_sections_flatten(node: SectionNode) -> List[SectionNode]:
    """Collect node and all descendants for scoring (efficient traversal)."""
    out: List[SectionNode] = [node]
    for ch in node.children:
        out.extend(_collect_sections_flatten(ch))
    return out


def navigate_pageindex(page_index: PageIndex, topic: str, top_k: int = 3) -> List[SectionNode]:
    """
    PageIndex query: scores all sections by keyword overlap with topic (including
    nested children), returns top_k sections. Efficient and intuitive traversal.
    """
    topic_lower = topic.lower()
    candidates = _collect_sections_flatten(page_index.root)
    scored = []
    for sec in candidates:
        text = f"{sec.title}\n{sec.summary or ''}\n{' '.join(sec.key_entities)}".lower()
        score = sum(1 for tok in topic_lower.split() if len(tok) > 1 and tok in text)
        scored.append((score, sec))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for score, sec in scored[:top_k] if score > 0]

