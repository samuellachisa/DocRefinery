from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Optional, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine, func
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select

from src.llm import LLMConfig, call_text_llm
from src.models import LDU, PageIndex, ProvenanceChain, ProvenanceSpan, SectionNode


@dataclass
class VectorStoreConfig:
    persist_directory: str = ".refinery/chroma"


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


class FactTable:
    """Simple SQLite-backed fact table for numerical/financial facts."""

    def __init__(self, db_path: Path = Path(".refinery/facts.sqlite")) -> None:
        self.engine: Engine = create_engine(f"sqlite:///{db_path}")
        self.metadata = MetaData()
        self.facts_table = Table(
            "facts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("doc_id", String, index=True),
            Column("entity", String),
            Column("metric", String),
            Column("value", Float),
            Column("period", String),
            Column("page_number", Integer),
            Column("content_hash", String),
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

    def query_metric(self, doc_id: str, metric: str) -> List[dict]:
        """
        Fuzzy metric lookup:
        - Filters by doc_id.
        - Matches rows where lower(metric) contains the provided metric substring.
        """
        metric_norm = metric.lower()
        with self.engine.begin() as conn:
            stmt = (
                select(self.facts_table)
                .where(self.facts_table.c.doc_id == doc_id)
                .where(func.lower(self.facts_table.c.metric).like(f"%{metric_norm}%"))
            )
            rows = conn.execute(stmt).mappings().all()
        return [dict(r) for r in rows]


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
        # Use first page_ref and bbox as provenance
        page_number = best.page_refs[0] if best.page_refs else 1
        span = ProvenanceSpan(
            doc_name=doc_name,
            doc_id=best.doc_id,
            page_number=page_number,
            bbox=best.bounding_box,
            content_hash=best.content_hash,
            snippet=best.content[:300],
        )
        chain = ProvenanceChain(spans=[span])
        return best.content, chain

    # --- audit mode ---

    def audit_claim(self, claim: str, doc_name: str) -> tuple[str, ProvenanceChain, bool]:
        """
        Audit mode: multi-source verification. Retrieves multiple candidate chunks;
        if at least one provides support, returns verified=True with full provenance.
        Otherwise explicitly flags as unverifiable with clear message.
        """
        matches = self.semantic_search(claim, k=5)
        if not matches:
            return (
                "Claim could not be verified: no supporting content found in the document. Flagged as unverifiable.",
                ProvenanceChain(spans=[]),
                False,
            )
        # Build provenance from all supporting chunks (multi-source)
        spans: List[ProvenanceSpan] = []
        for best in matches:
            page_number = best.page_refs[0] if best.page_refs else 1
            spans.append(
                ProvenanceSpan(
                    doc_name=doc_name,
                    doc_id=best.doc_id,
                    page_number=page_number,
                    bbox=best.bounding_box,
                    content_hash=best.content_hash,
                    snippet=best.content[:300],
                )
            )
        chain = ProvenanceChain(spans=spans)
        supporting = "\n\n".join(m.content[:500] for m in matches[:3])
        return supporting, chain, True


class AgentState(TypedDict, total=False):
    """LangGraph agent state for querying a single document."""

    question: str
    doc_name: str
    answer: str
    provenance: ProvenanceChain
    use_structured: bool
    chosen_tool: str


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
                    bbox=None,
                    content_hash=section_hash,
                    snippet=summary_text or "",
                )])
                return state
        else:
            matches = agent.semantic_search(question, k=1)
            if matches:
                best = matches[0]
                page_number = best.page_refs[0] if best.page_refs else 1
                state["answer"] = best.content
                state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                    doc_name=doc_name,
                    doc_id=best.doc_id,
                    page_number=page_number,
                    bbox=best.bounding_box,
                    content_hash=best.content_hash,
                    snippet=best.content[:300],
                )])
                return state

        # Fallback: try other tools
        if chosen != "structured_query":
            rows = agent.structured_query(agent.page_index.doc_id, "amount")
            if rows:
                total = sum(r.get("value", 0.0) for r in rows)
                set_structured(rows, f"Found numerical data: total across {len(rows)} rows: {total}.")
                return state
        if chosen != "semantic_search":
            matches = agent.semantic_search(question, k=1)
            if matches:
                best = matches[0]
                page_number = best.page_refs[0] if best.page_refs else 1
                state["answer"] = best.content
                state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                    doc_name=doc_name, doc_id=best.doc_id, page_number=page_number,
                    bbox=best.bounding_box, content_hash=best.content_hash, snippet=best.content[:300],
                )])
                return state
        if chosen != "pageindex_navigate":
            sections = agent.pageindex_navigate(question, top_k=1)
            if sections:
                sec = sections[0]
                state["answer"] = sec.summary or "Relevant section found in PageIndex."
                state["provenance"] = ProvenanceChain(spans=[ProvenanceSpan(
                    doc_name=doc_name, doc_id=agent.page_index.doc_id, page_number=sec.page_start,
                    bbox=None, content_hash=hashlib.sha256((sec.summary or "").encode()).hexdigest()[:32], snippet=sec.summary or "",
                )])
                return state

        state["answer"] = "No relevant information found."
        state["provenance"] = ProvenanceChain(spans=[])
        return state

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("router")
    graph.add_edge("router", "answer")
    graph.add_edge("answer", END)
    return graph.compile()

