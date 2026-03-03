from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, TypedDict

import chromadb
from chromadb.utils import embedding_functions
from langgraph.graph import StateGraph, END
from sqlalchemy import Column, Float, Integer, MetaData, String, Table, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select

from src.models import LDU, PageIndex, ProvenanceChain, ProvenanceSpan


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
        with self.engine.begin() as conn:
            stmt = (
                select(self.facts_table)
                .where(self.facts_table.c.doc_id == doc_id)
                .where(self.facts_table.c.metric == metric)
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

    def pageindex_navigate(self, topic: str) -> List[PageIndex]:
        # Current simple implementation: always return the stored PageIndex
        return [self.page_index]

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
        Audit mode:
        - Attempts to find support for a claim via semantic_search.
        - If a relevant chunk is found, returns it with provenance and verified=True.
        - Otherwise returns unverifiable.
        """
        matches = self.semantic_search(claim, k=1)
        if not matches:
            return (
                "Claim not found or unverifiable based on available chunks.",
                ProvenanceChain(spans=[]),
                False,
            )

        best = matches[0]
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
        return best.content, chain, True


class AgentState(TypedDict, total=False):
    """LangGraph agent state for querying a single document."""

    question: str
    doc_name: str
    answer: str
    provenance: ProvenanceChain
    use_structured: bool


def build_langgraph_agent(agent: QueryAgent):
    """
    Build a minimal LangGraph agent that uses three conceptual tools:
    - pageindex_navigate (via agent.pageindex_navigate)
    - semantic_search (via agent.semantic_search)
    - structured_query (via agent.structured_query)

    For now, orchestration is simple: we prioritise semantic search and fall
    back to PageIndex, but the structure is ready for richer routing.
    """

    def answer_node(state: AgentState) -> AgentState:
        question = state["question"]
        doc_name = state.get("doc_name", "document")

        # Decide whether to use structured_query (for numeric/metric questions)
        lower_q = question.lower()
        numeric_keywords = ("revenue", "income", "profit", "loss", "amount", "million", "billion", "$")
        if any(k in lower_q for k in numeric_keywords):
            state["use_structured"] = True
        else:
            state["use_structured"] = False

        # Try structured_query first when appropriate
        if state["use_structured"]:
            # Heuristic: look for metric keyword and query fact table
            metric = next((k for k in numeric_keywords if k in lower_q), "revenue")
            rows = agent.structured_query(agent.page_index.doc_id, metric)
            if rows:
                best_row = rows[0]
                answer_txt = (
                    f"{best_row['metric']} for {best_row['entity']} "
                    f"in {best_row['period']}: {best_row['value']}"
                )
                page_number = best_row.get("page_number", 1)
                span = ProvenanceSpan(
                    doc_name=doc_name,
                    doc_id=agent.page_index.doc_id,
                    page_number=page_number,
                    bbox=None,
                    content_hash=best_row.get("content_hash") or None,
                    snippet=answer_txt,
                )
                state["answer"] = answer_txt
                state["provenance"] = ProvenanceChain(spans=[span])
                return state

        # Use semantic_search as primary tool otherwise
        matches = agent.semantic_search(question, k=1)
        if matches:
            best = matches[0]
            page_number = best.page_refs[0] if best.page_refs else 1
            span = ProvenanceSpan(
                doc_name=doc_name,
                doc_id=best.doc_id,
                page_number=page_number,
                bbox=best.bounding_box,
                content_hash=best.content_hash,
                snippet=best.content[:300],
            )
            state["answer"] = best.content
            state["provenance"] = ProvenanceChain(spans=[span])
            return state

        # Fallback: use PageIndex navigation(query) to at least constrain the section
        from src.agents.indexer import navigate_pageindex  # local import to avoid cycle

        sections = navigate_pageindex(agent.page_index, question, top_k=1)
        if sections:
            sec = sections[0]
            span = ProvenanceSpan(
                doc_name=doc_name,
                doc_id=agent.page_index.doc_id,
                page_number=sec.page_start,
                bbox=None,
                content_hash=None,
                snippet=sec.summary or "",
            )
            state["answer"] = sec.summary or "Relevant section found in PageIndex."
            state["provenance"] = ProvenanceChain(spans=[span])
            return state

        state["answer"] = "No relevant information found."
        state["provenance"] = ProvenanceChain(spans=[])
        return state

    graph = StateGraph(AgentState)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()

