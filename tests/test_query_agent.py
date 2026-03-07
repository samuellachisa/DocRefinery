"""
Tests for the LangGraph-based QueryAgent: routing intent (navigation vs semantic vs numeric),
graceful fallbacks when the chosen tool returns weak/empty results, and structured answer format.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.query_agent import (
    QueryAgent,
    build_langgraph_agent,
    invoke_structured,
    state_to_structured_response,
)
from src.models import BoundingBox, LDU
from src.models.ldu import ChunkType
from src.models.page_index import PageIndex, SectionNode


def _minimal_page_index(doc_id: str = "test-doc") -> PageIndex:
    root = SectionNode(
        id="root",
        title="Document",
        page_start=1,
        page_end=2,
        summary="Root section.",
    )
    return PageIndex(doc_id=doc_id, root=root)


def _minimal_ldu(doc_id: str = "test-doc") -> LDU:
    return LDU(
        id="ldu-1",
        doc_id=doc_id,
        chunk_type=ChunkType.paragraph,
        content="Sample content for semantic search.",
        page_refs=[1],
        bounding_box=BoundingBox(x0=0, y0=0, x1=100, y1=100),
        token_count=10,
        content_hash="abc123",
    )


@pytest.fixture
def temp_chroma(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture
def temp_fact_db(tmp_path: Path) -> Path:
    return tmp_path / "facts.sqlite"


@pytest.fixture
def query_agent(temp_chroma: Path, temp_fact_db: Path) -> QueryAgent:
    from src.agents.query_agent import FactTable, VectorStore, VectorStoreConfig

    page_index = _minimal_page_index()
    ldus = [_minimal_ldu(page_index.doc_id)]
    vector_store = VectorStore(VectorStoreConfig(persist_directory=str(temp_chroma)))
    fact_table = FactTable(db_path=temp_fact_db)
    return QueryAgent(
        page_index=page_index,
        ldus=ldus,
        vector_store=vector_store,
        fact_table=fact_table,
    )


# ---------- Routing: intent → tool choice ----------


def test_routing_numeric_intent_uses_structured_query(query_agent: QueryAgent) -> None:
    """Numeric question (revenue, total) should use structured_query and return numeric answer."""
    fake_rows = [
        {
            "entity": "Division A",
            "metric": "revenue",
            "period": "2023",
            "value": 100.0,
            "page_number": 1,
            "content_hash": "h1",
        }
    ]
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="structured_query"),
        patch.object(query_agent, "structured_query", return_value=fake_rows),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "What was the total revenue?", "doc_name": "report.pdf"}
        )
    assert state.get("tool_used") == "structured_query"
    assert state.get("fallback_used") is False
    assert "revenue" in state["answer"].lower() or "100" in state["answer"]
    assert state["provenance"].spans


def test_routing_navigation_intent_uses_pageindex(query_agent: QueryAgent) -> None:
    """Navigation question (which section, where is X) should use pageindex_navigate."""
    section = SectionNode(
        id="s1",
        title="Methodology",
        page_start=3,
        page_end=4,
        summary="Methodology section describes the approach.",
    )
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="pageindex_navigate"),
        patch.object(query_agent, "pageindex_navigate", return_value=[section]),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "Where is the methodology section?", "doc_name": "report.pdf"}
        )
    assert state.get("tool_used") == "pageindex_navigate"
    assert state.get("fallback_used") is False
    assert "Methodology" in state["answer"] or "approach" in state["answer"]
    assert state["provenance"].spans


def test_routing_semantic_intent_uses_semantic_search(query_agent: QueryAgent) -> None:
    """Factual/explanation question should use semantic_search."""
    ldu = _minimal_ldu(query_agent.page_index.doc_id)
    ldu.content = "Risk is defined as the probability of loss."
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="semantic_search"),
        patch.object(query_agent, "semantic_search", return_value=[ldu]),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "What does the report say about risk?", "doc_name": "report.pdf"}
        )
    assert state.get("tool_used") == "semantic_search"
    assert state.get("fallback_used") is False
    assert "risk" in state["answer"].lower() or "probability" in state["answer"].lower()
    assert state["provenance"].spans


def test_keyword_fallback_numeric_question() -> None:
    """When LLM fails, keyword fallback routes numeric keywords to structured_query."""
    from src.agents.query_agent import _llm_choose_tool

    with patch("src.agents.query_agent.call_text_llm", side_effect=Exception("LLM down")):
        assert _llm_choose_tool("What was the total revenue?", None) == "structured_query"
        assert _llm_choose_tool("How much profit in millions?", None) == "structured_query"


def test_keyword_fallback_navigation_question() -> None:
    """When LLM fails, keyword fallback routes structure/section keywords to pageindex_navigate."""
    from src.agents.query_agent import _llm_choose_tool

    with patch("src.agents.query_agent.call_text_llm", side_effect=Exception("LLM down")):
        assert _llm_choose_tool("Where is the executive summary?", None) == "pageindex_navigate"
        assert _llm_choose_tool("Which section has the table of contents?", None) == "pageindex_navigate"


def test_keyword_fallback_semantic_question() -> None:
    """When LLM fails, keyword fallback routes generic questions to semantic_search."""
    from src.agents.query_agent import _llm_choose_tool

    with patch("src.agents.query_agent.call_text_llm", side_effect=Exception("LLM down")):
        assert _llm_choose_tool("What is the main finding?", None) == "semantic_search"


# ---------- Graceful fallbacks when chosen tool returns weak/empty ----------


def test_fallback_when_primary_returns_empty(query_agent: QueryAgent) -> None:
    """When chosen tool (structured_query) returns [], fallback to semantic_search succeeds."""
    ldu = _minimal_ldu(query_agent.page_index.doc_id)
    ldu.content = "Fallback content from semantic search."
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="structured_query"),
        patch.object(query_agent, "structured_query", return_value=[]),
        patch.object(query_agent, "semantic_search", return_value=[ldu]),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "What was the revenue?", "doc_name": "report.pdf"}
        )
    assert state.get("tool_used") == "semantic_search"
    assert state.get("fallback_used") is True
    assert "Fallback content" in state["answer"]


def test_fallback_when_all_tools_empty(query_agent: QueryAgent) -> None:
    """When all tools return no results, answer is clear and answer_reason is no_results."""
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="semantic_search"),
        patch.object(query_agent, "semantic_search", return_value=[]),
        patch.object(query_agent, "structured_query", return_value=[]),
        patch.object(query_agent, "pageindex_navigate", return_value=[]),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "What is the meaning of life?", "doc_name": "report.pdf"}
        )
    assert state["answer"] == "No relevant information found."
    assert state.get("tool_used") is None
    assert state.get("fallback_used") is False
    assert state.get("answer_reason") == "no_results"
    assert len(state["provenance"].spans) == 0


def test_fallback_from_pageindex_to_structured(query_agent: QueryAgent) -> None:
    """When pageindex_navigate returns [], fallback can use structured_query."""
    fake_rows = [{"entity": "E", "metric": "amount", "period": "2023", "value": 50.0, "page_number": 1, "content_hash": "x"}]
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="pageindex_navigate"),
        patch.object(query_agent, "pageindex_navigate", return_value=[]),
        patch.object(query_agent, "structured_query", return_value=fake_rows),
    ):
        graph = build_langgraph_agent(query_agent)
        state = graph.invoke(
            {"question": "Where is the summary?", "doc_name": "report.pdf"}
        )
    assert state.get("tool_used") == "structured_query"
    assert state.get("fallback_used") is True
    assert "50" in state["answer"] or "numerical" in state["answer"].lower() or "total" in state["answer"].lower()


# ---------- Structured answer format (JSON) for downstream integration ----------


def test_state_to_structured_response_has_required_keys() -> None:
    """state_to_structured_response returns JSON-serializable dict with expected keys."""
    from src.models import ProvenanceChain, ProvenanceSpan

    state = {
        "answer": "Test answer.",
        "provenance": ProvenanceChain(spans=[
            ProvenanceSpan(
                doc_name="doc.pdf",
                doc_id="id1",
                page_number=1,
                content_hash="h1",
                snippet="Snippet.",
            )
        ]),
        "tool_used": "semantic_search",
        "fallback_used": False,
        "answer_reason": None,
        "chosen_tool": "semantic_search",
    }
    out = state_to_structured_response(state)
    assert "answer" in out
    assert "provenance" in out
    assert "tool_used" in out
    assert "fallback_used" in out
    assert "answer_reason" in out
    assert "chosen_tool" in out
    assert out["answer"] == "Test answer."
    assert out["tool_used"] == "semantic_search"
    assert out["fallback_used"] is False
    assert isinstance(out["provenance"], list)
    assert len(out["provenance"]) == 1
    assert out["provenance"][0]["page_number"] == 1
    assert out["provenance"][0]["snippet"] == "Snippet."


def test_structured_response_no_results_shape() -> None:
    """Structured response when no results has answer_reason and null tool_used."""
    state = {
        "answer": "No relevant information found.",
        "provenance": None,
        "tool_used": None,
        "fallback_used": False,
        "answer_reason": "no_results",
        "chosen_tool": "semantic_search",
    }
    # state_to_structured_response expects provenance to be ProvenanceChain or we need to handle None
    from src.models import ProvenanceChain

    state["provenance"] = ProvenanceChain(spans=[])
    out = state_to_structured_response(state)
    assert out["answer_reason"] == "no_results"
    assert out["tool_used"] is None
    assert out["provenance"] == []


def test_invoke_structured_returns_json_serializable_dict(
    query_agent: QueryAgent,
) -> None:
    """invoke_structured returns a dict that can be serialized to JSON for downstream systems."""
    ldu = _minimal_ldu(query_agent.page_index.doc_id)
    ldu.content = "Structured answer content."
    with (
        patch("src.agents.query_agent._llm_choose_tool", return_value="semantic_search"),
        patch.object(query_agent, "semantic_search", return_value=[ldu]),
    ):
        result = invoke_structured(
            query_agent,
            question="What is the content?",
            doc_name="report.pdf",
        )
    assert isinstance(result, dict)
    assert "answer" in result
    assert "provenance" in result
    assert "tool_used" in result
    assert "fallback_used" in result
    assert result["tool_used"] == "semantic_search"
    assert "Structured answer content" in result["answer"]
    import json
    json_str = json.dumps(result)
    assert "Structured answer content" in json_str
