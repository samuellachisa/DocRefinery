from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Optional, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from sqlalchemy import Column, Float, Index, Integer, MetaData, String, Table, create_engine, func
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select

from src.llm import LLMConfig, call_text_llm
from src.models import LDU, PageIndex, ProvenanceChain, ProvenanceSpan, SectionNode


@dataclass
class VectorStoreConfig:
    persist_directory: str = ".refinery/chroma"


@dataclass
class AuditResult:
    """Result of audit_claim with explicit confidence and contradiction handling."""

    supporting_text: str
    provenance: ProvenanceChain
    verified: bool
    confidence: Literal["high", "low", "none"]
    contradiction_detected: bool


class VectorStore:
    def __init__(self, cfg: VectorStoreConfig) -> None:
        self.client = chromadb.PersistentClient(path=cfg.persist_directory)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="ldu_chunks", embedding_function=self.embedding_fn
        )

    def ingest(self, ldus: List[LDU]) -> None:
        ids = [ldu.id for ldu in ldus]
        texts = [ldu.content for ldu in ldus]
        metadatas = [
            {
                "doc_id": ldu.doc_id,
                "page_refs": ldu.page_refs,
                "chunk_type": ldu.chunk_type.value,
                "content_hash": ldu.content_hash,
                "parent_section": ldu.parent_section or "",
            }
            for ldu in ldus
        ]
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def ingest_batch(self, ldus: List[LDU], batch_size: int = 500) -> None:
        """
        Ingest LDUs in batches for large corpora to avoid memory spikes and timeouts.
        Preserves metadata (doc_id, page_refs, chunk_type, content_hash, parent_section).
        """
        for i in range(0, len(ldus), batch_size):
            chunk = ldus[i : i + batch_size]
            self.ingest(chunk)

    def query(self, query: str, k: int = 5) -> List[Tuple[str, dict]]:
        res = self.collection.query(query_texts=[query], n_results=k)
        matches: List[Tuple[str, dict]] = []
        for ids, metas, docs in zip(
            res.get("ids", [[]]), res.get("metadatas", [[]]), res.get("documents", [[]])
        ):
            for i, m, d in zip(ids, metas, docs):
                m = dict(m or {})
                m["content"] = d
                matches.append((i, m))
        return matches

    def query_with_distances(
        self, query: str, k: int = 5
    ) -> List[Tuple[str, dict, float]]:
        """Like query() but also returns similarity distance per result (L2; lower is more similar)."""
        res = self.collection.query(
            query_texts=[query], n_results=k, include=["metadatas", "documents", "distances"]
        )
        out: List[Tuple[str, dict, float]] = []
        for ids, metas, docs, dists in zip(
            res.get("ids", [[]]),
            res.get("metadatas", [[]]),
            res.get("documents", [[]]),
            res.get("distances", [[]]),
        ):
            for i, m, d, dist in zip(ids, metas, docs, dists):
                m = dict(m or {})
                m["content"] = d
                out.append((i, m, float(dist)))
        return out


class FactTable:
    """
    SQLite-backed fact table for numerical/financial facts with indexing for
    optimized queries over large corpora (doc_id, entity, period, content_hash).
    """

    def __init__(self, db_path: Path = Path(".refinery/facts.sqlite")) -> None:
        self.engine: Engine = create_engine(f"sqlite:///{db_path}")
        self.metadata = MetaData()
        self.facts_table = Table(
            "facts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("doc_id", String, index=True),
            Column("entity", String, index=True),
            Column("metric", String),
            Column("value", Float),
            Column("period", String, index=True),
            Column("page_number", Integer),
            Column("content_hash", String, index=True),
            Index("ix_facts_doc_entity", "doc_id", "entity"),
            Index("ix_facts_doc_period", "doc_id", "period"),
            Index("ix_facts_doc_metric", "doc_id", "metric"),
        )
        self.metadata.create_all(self.engine)

    def insert_fact(
        self,
        doc_id: str,
        entity: str,
        metric: str,
        value: float,
        period: str,
        page_number: int,
        content_hash: str,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.facts_table.insert().values(
                    doc_id=doc_id,
                    entity=entity,
                    metric=metric,
                    value=value,
                    period=period,
                    page_number=page_number,
                    content_hash=content_hash,
                )
            )

    def query_metric(
        self, doc_id: str, metric: str, limit: Optional[int] = None, offset: int = 0
    ) -> List[dict]:
        """
        Fuzzy metric lookup (indexed by doc_id, metric):
        - Filters by doc_id.
        - Matches rows where lower(metric) contains the provided metric substring.
        - limit/offset support for large result sets.
        """
        metric_norm = metric.lower()
        with self.engine.begin() as conn:
            stmt = (
                select(self.facts_table)
                .where(self.facts_table.c.doc_id == doc_id)
                .where(func.lower(self.facts_table.c.metric).like(f"%{metric_norm}%"))
                .offset(offset)
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def query_by_entity(
        self, doc_id: str, entity: str, limit: Optional[int] = None
    ) -> List[dict]:
        """Return all facts for a given document and entity (uses ix_facts_doc_entity)."""
        with self.engine.begin() as conn:
            stmt = (
                select(self.facts_table)
                .where(self.facts_table.c.doc_id == doc_id)
                .where(self.facts_table.c.entity == entity)
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def query_by_period_range(
        self,
        doc_id: str,
        period_start: Optional[str] = None,
        period_end: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        """
        Return facts within a period range (uses ix_facts_doc_period).
        period_start/period_end are inclusive; string comparison (e.g. "2022" <= period <= "2024").
        """
        with self.engine.begin() as conn:
            stmt = select(self.facts_table).where(self.facts_table.c.doc_id == doc_id)
            if period_start is not None:
                stmt = stmt.where(self.facts_table.c.period >= period_start)
            if period_end is not None:
                stmt = stmt.where(self.facts_table.c.period <= period_end)
            if limit is not None:
                stmt = stmt.limit(limit)
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]

    def query_by_content_hash(self, content_hash: str) -> List[dict]:
        """Return all fact rows originating from the given table content_hash (provenance lookup)."""
        with self.engine.begin() as conn:
            stmt = select(self.facts_table).where(
                self.facts_table.c.content_hash == content_hash
            )
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]


def _span_from_ldu(ldu: LDU, doc_name: str) -> ProvenanceSpan:
    """Build a ProvenanceSpan from an LDU; sets page_end when LDU spans multiple pages."""
    page_number = ldu.page_refs[0] if ldu.page_refs else 1
    page_end: Optional[int] = None
    if ldu.page_refs and len(ldu.page_refs) > 1:
        page_end = ldu.page_refs[-1]
    return ProvenanceSpan(
        doc_name=doc_name,
        doc_id=ldu.doc_id,
        page_number=page_number,
        page_end=page_end,
        bbox=ldu.bounding_box,
        content_hash=ldu.content_hash,
        snippet=ldu.content[:300],
    )


class QueryAgent:
    """
    Query Interface Agent.

    Tools:
    - pageindex_navigate: navigate PageIndex tree for topical sections.
    - semantic_search: vector retrieval over LDUs.
    - structured_query: SQL over FactTable.
    """

    def __init__(
        self,
        page_index: PageIndex,
        ldus: List[LDU],
        vector_store: VectorStore | None = None,
        fact_table: FactTable | None = None,
    ) -> None:
        self.page_index = page_index
        self.ldus = {ldu.id: ldu for ldu in ldus}
        self.vector_store = vector_store or VectorStore(VectorStoreConfig())
        self.fact_table = fact_table or FactTable()
        # Ensure LDUs are ingested
        self.vector_store.ingest(ldus)

    # --- tools ---

    def pageindex_navigate(self, topic: str, top_k: int = 5) -> List[SectionNode]:
        """Navigate the PageIndex by topic; returns the top_k section nodes most relevant to the topic."""
        from src.agents.indexer import navigate_pageindex

        return navigate_pageindex(self.page_index, topic, top_k=top_k)

    def semantic_search(self, query: str, k: int = 5) -> List[LDU]:
        matches = self.vector_store.query(query, k=k)
        results: List[LDU] = []
        for ldu_id, meta in matches:
            ldu = self.ldus.get(ldu_id)
            if ldu:
                results.append(ldu)
        return results

    def structured_query(self, doc_id: str, metric: str) -> List[dict]:
        return self.fact_table.query_metric(doc_id, metric)

    # --- high-level answering ---

    def answer(self, question: str, doc_name: str) -> tuple[str, ProvenanceChain]:
        """
        Very simple answering strategy:
        - Use semantic_search over LDUs.
        - Take the top match's content as the answer.
        - Build a ProvenanceChain from the LDU metadata.
        """
        matches = self.semantic_search(question, k=1)
        if not matches:
            return "No relevant information found.", ProvenanceChain(spans=[])

        best = matches[0]
        span = _span_from_ldu(best, doc_name)
        chain = ProvenanceChain(spans=[span])
        return best.content, chain

    # --- audit mode ---

    def audit_claim(
        self,
        claim: str,
        doc_name: str,
        confidence_distance_threshold: float = 1.0,
    ) -> "AuditResult":
        """
        Audit mode: multi-source verification with explicit handling of
        low-confidence matches and contradictory evidence.
        - Uses semantic search with distances; marks confidence as "low" when
          best match distance is above confidence_distance_threshold (L2; lower = more similar).
        - Flags contradiction_detected when any retrieved snippet contains
          phrases that often indicate contradicting evidence (e.g. "contrary to", "does not support").
        """
        raw = self.vector_store.query_with_distances(claim, k=5)
        matches_with_dist: List[Tuple[LDU, float]] = []
        for ldu_id, meta, dist in raw:
            ldu = self.ldus.get(ldu_id)
            if ldu:
                matches_with_dist.append((ldu, dist))
        if not matches_with_dist:
            return AuditResult(
                supporting_text="Claim could not be verified: no supporting content found in the document. Flagged as unverifiable.",
                provenance=ProvenanceChain(spans=[]),
                verified=False,
                confidence="none",
                contradiction_detected=False,
            )
        matches = [m for m, _ in matches_with_dist]
        best_distance = matches_with_dist[0][1]
        confidence: Literal["high", "low", "none"] = (
            "low" if best_distance > confidence_distance_threshold else "high"
        )
        # Contradiction heuristic: any snippet contains common contradiction phrases
        contradiction_phrases = re.compile(
            r"\b(contrary to|does not support|do not support|contradicts?)\b",
            re.IGNORECASE,
        )
        contradiction_detected = any(
            contradiction_phrases.search(m.content[:500]) for m in matches[:3]
        )
        spans = [_span_from_ldu(m, doc_name) for m in matches]
        chain = ProvenanceChain(spans=spans)
        supporting = "\n\n".join(m.content[:500] for m in matches[:3])
        return AuditResult(
            supporting_text=supporting,
            provenance=chain,
            verified=True,
            confidence=confidence,
            contradiction_detected=contradiction_detected,
        )


class AgentState(TypedDict, total=False):
    """LangGraph agent state for querying a single document."""

    question: str
    doc_name: str
    answer: str
    provenance: ProvenanceChain
    use_structured: bool
    chosen_tool: str
    tool_used: Optional[str]  # tool that actually produced the answer
    fallback_used: bool  # True if primary tool returned nothing and a fallback did
    answer_reason: Optional[str]  # e.g. "no_results" when no tool had data


def _llm_choose_tool(question: str, llm_cfg: LLMConfig) -> Literal["semantic_search", "structured_query", "pageindex_navigate"]:
    """Dynamic tool selection: LLM picks the best tool for the question."""
    prompt = (
        "You must choose exactly one tool to answer this question.\n"
        "Tools:\n"
        "- pageindex_navigate: for document structure, table of contents, which section or page contains a topic.\n"
        "- semantic_search: for factual content, definitions, explanations, general content.\n"
        "- structured_query: for numbers, metrics, revenue, totals, amounts, financial data.\n\n"
        "Question: " + question + "\n\n"
        "Reply with only one word: pageindex_navigate, semantic_search, or structured_query."
    )
    try:
        raw = call_text_llm(prompt, llm_cfg).strip().lower()
        if "pageindex" in raw or "navigate" in raw:
            return "pageindex_navigate"
        if "structured" in raw or "query" in raw:
            return "structured_query"
        return "semantic_search"
    except Exception:
        lower = question.lower()
        if any(k in lower for k in ("revenue", "income", "profit", "amount", "total", "million", "billion", "$")):
            return "structured_query"
        if any(k in lower for k in ("section", "page", "where", "structure", "contents", "toc")):
            return "pageindex_navigate"
        return "semantic_search"


def build_langgraph_agent(agent: QueryAgent):
    """
    LangGraph agent with dynamic LLM-based tool selection. Every answer
    has full provenance (doc_name, page_number, bbox, content_hash).
    """

    llm_cfg = LLMConfig.from_env()

    def router_node(state: AgentState) -> AgentState:
        state["chosen_tool"] = _llm_choose_tool(state["question"], llm_cfg)
        return state

    def answer_node(state: AgentState) -> AgentState:
        question = state["question"]
        doc_name = state.get("doc_name", "document")
        chosen = state.get("chosen_tool") or "semantic_search"
        lower_q = question.lower()

        def set_structured(rows: list, answer_txt: str) -> None:
            first = rows[0]
            page_number = first.get("page_number", 1)
            fact_hash = first.get("content_hash") or hashlib.sha256(
                f"{first.get('entity')}:{first.get('metric')}:{page_number}".encode()
            ).hexdigest()[:32]
            state["answer"] = answer_txt
            state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                doc_name=doc_name,
                doc_id=agent.page_index.doc_id,
                page_number=page_number,
                bbox=None,
                content_hash=fact_hash,
                snippet=answer_txt,
            )])

        # Try chosen tool first
        if chosen == "structured_query":
            metric = next(
                (k for k in ("revenue", "income", "profit", "amount", "total", "sum", "million", "billion") if k in lower_q),
                "amount",
            )
            rows = agent.structured_query(agent.page_index.doc_id, metric)
            if rows:
                total = sum(r.get("value", 0.0) for r in rows)
                examples = [f"{r.get('entity')} – {r.get('metric')} {r.get('period')}: {r.get('value')}" for r in rows[:3]]
                set_structured(rows, f"Total {metric} across {len(rows)} fact rows: {total}. Examples: {'; '.join(examples)}")
                state["tool_used"] = "structured_query"
                state["fallback_used"] = False
                return state
        elif chosen == "pageindex_navigate":
            sections = agent.pageindex_navigate(question, top_k=1)
            if sections:
                sec = sections[0]
                summary_text = sec.summary or ""
                section_hash = hashlib.sha256(summary_text.encode()).hexdigest()[:32] if summary_text else "pageindex"
                state["answer"] = sec.summary or "Relevant section found in PageIndex."
                state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                    doc_name=doc_name,
                    doc_id=agent.page_index.doc_id,
                    page_number=sec.page_start,
                    page_end=sec.page_end if sec.page_end != sec.page_start else None,
                    bbox=None,
                    content_hash=section_hash,
                    snippet=summary_text or "",
                )])
                state["tool_used"] = "pageindex_navigate"
                state["fallback_used"] = False
                return state
        else:
            matches = agent.semantic_search(question, k=1)
            if matches:
                best = matches[0]
                state["answer"] = best.content
                state["provenance"] = ProvenanceChain(spans=[_span_from_ldu(best, doc_name)])
                state["tool_used"] = "semantic_search"
                state["fallback_used"] = False
                return state

        # Fallback: chosen tool returned nothing; try other tools
        if chosen != "structured_query":
            rows = agent.structured_query(agent.page_index.doc_id, "amount")
            if rows:
                total = sum(r.get("value", 0.0) for r in rows)
                set_structured(rows, f"Found numerical data: total across {len(rows)} rows: {total}.")
                state["tool_used"] = "structured_query"
                state["fallback_used"] = True
                return state
        if chosen != "semantic_search":
            matches = agent.semantic_search(question, k=1)
            if matches:
                best = matches[0]
                state["answer"] = best.content
                state["provenance"] = ProvenanceChain(spans=[_span_from_ldu(best, doc_name)])
                state["tool_used"] = "semantic_search"
                state["fallback_used"] = True
                return state
        if chosen != "pageindex_navigate":
            sections = agent.pageindex_navigate(question, top_k=1)
            if sections:
                sec = sections[0]
                state["answer"] = sec.summary or "Relevant section found in PageIndex."
                state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                    doc_name=doc_name, doc_id=agent.page_index.doc_id, page_number=sec.page_start,
                    page_end=sec.page_end if sec.page_end != sec.page_start else None,
                    bbox=None, content_hash=hashlib.sha256((sec.summary or "").encode()).hexdigest()[:32], snippet=sec.summary or "",
                )])
                state["tool_used"] = "pageindex_navigate"
                state["fallback_used"] = True
                return state

        # All tools returned no usable result
        state["answer"] = "No relevant information found."
        state["provenance"] = ProvenanceChain(spans=[])
        state["tool_used"] = None
        state["fallback_used"] = False
        state["answer_reason"] = "no_results"
        return state

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def state_to_structured_response(state: AgentState) -> dict:
    """
    Convert final agent state to a JSON-serializable dict for downstream integration.
    Keys: answer, provenance (list of span dicts), tool_used, fallback_used, answer_reason, chosen_tool.
    """
    provenance = state.get("provenance")
    if provenance is not None:
        provenance_payload = [s.model_dump(mode="json") for s in provenance.spans]
    else:
        provenance_payload = []
    return {
        "answer": state.get("answer", ""),
        "provenance": provenance_payload,
        "tool_used": state.get("tool_used"),
        "fallback_used": state.get("fallback_used", False),
        "answer_reason": state.get("answer_reason"),
        "chosen_tool": state.get("chosen_tool"),
    }


def invoke_structured(
    agent: QueryAgent,
    question: str,
    doc_name: str = "document",
    config: Optional[dict] = None,
) -> dict:
    """
    Run the LangGraph agent and return a structured JSON-serializable response
    for downstream systems (answer, provenance, tool_used, fallback_used, etc.).
    """
    graph = build_langgraph_agent(agent)
    final_state = graph.invoke(
        {"question": question, "doc_name": doc_name},
        config=config or {"configurable": {"run_name": "docrefinery_query"}},
    )
    return state_to_structured_response(final_state)

