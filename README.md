## Document Intelligence Refinery

This repository implements **The Document Intelligence Refinery** – a production-inspired, multi-stage agentic pipeline for converting heterogeneous enterprise documents into structured, queryable, spatially-indexed knowledge.

### Features (High Level)

- **Triage Agent (`src/agents/triage.py`)**: Profiles each document (origin type, layout complexity, language, domain hint, estimated extraction cost) using configurable thresholds and domain keywords from `rubric/extraction_rules.yaml`, then writes `DocumentProfile` JSON to `.refinery/profiles/`.
- **Multi-Strategy Extraction (`src/agents/extractor.py`, `src/strategies/`)**:
  - Fast text extraction (pdfplumber) with confidence scoring.
  - Layout-aware extraction (Docling) for complex/table-heavy documents.
  - Vision-augmented extraction (VLM via HTTP API) for scanned/low-confidence pages, with budget caps from config.
  - Escalation guard and extraction ledger in `.refinery/extraction_ledger.jsonl`.
  - Cached `ExtractedDocument` JSON snapshots in `.refinery/extracted/` so downstream stages can be re-run without re-extracting.
- **Semantic Chunking & PageIndex**:
  - Normalized `ExtractedDocument` model and `LDU` chunks with stable `content_hash` identifiers.
  - Config-driven chunking rules (token budgets, list handling) from `rubric/extraction_rules.yaml`.
  - PageIndex tree to navigate long documents before vector search.
- **Query Agent & Storage**:
  - LangGraph-based agent with `pageindex_navigate`, `semantic_search`, and `structured_query` tools.
  - Vector store (Chroma) for LDU embeddings in `.refinery/chroma/`.
  - Structured `FactTable` for numeric facts in `.refinery/facts.sqlite`.
  - Every answer includes a `ProvenanceChain` (document, page, bbox, content_hash).

### Project Layout

- `src/models/` – Core Pydantic models:
  - `document_profile.py` – `DocumentProfile` and enums.
  - `extracted_document.py` – normalized extraction schema.
  - `ldu.py` – logical document units.
  - `page_index.py` – PageIndex tree structures.
  - `provenance.py` – provenance spans and chains.
- `src/agents/` – Pipeline agents:
  - `triage.py` – Triage Agent.
  - `extractor.py` – ExtractionRouter.
  - `chunker.py` – Semantic Chunking Engine (Phase 3).
  - `indexer.py` – PageIndex builder (Phase 3).
  - `query_agent.py` – LangGraph query interface (Phase 4).
- `src/strategies/` – Extraction strategies:
  - `base.py` – shared strategy interface.
  - `fast_text.py` – pdfplumber-based extractor with confidence scoring.
  - `layout_docling.py` – Docling-based layout-aware extractor.
  - `vision_vlm.py` – VLM-based extractor with budget guard.
- `rubric/extraction_rules.yaml` – Externalized thresholds, domain keywords, and chunking rules.
- `.refinery/` – Runtime artifacts:
  - `profiles/` – `DocumentProfile` JSON outputs.
  - `extracted/` – cached `ExtractedDocument` JSON snapshots.
  - `pageindex/` – PageIndex JSON trees.
  - `extraction_ledger.jsonl` – extraction trace and cost estimates.
  - `chroma/` – persisted vector store for LDUs.
  - `facts.sqlite` – SQLite-backed fact table for numeric/financial facts.
- `scripts/` – Utility entrypoints:
  - `run_pipeline.py` – programmatic end-to-end pipeline runner.
  - `vector_explorer.py` – CLI explorer for the vector store (LDUs).
  - `chunk_from_extracted.py` – run chunking + ingest from cached extraction only.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you are not using editable installs:

```bash
pip install .
```

### Quickstart – `run_pipeline.sh`

For most use cases, use the interactive shell wrapper:

```bash
./run_pipeline.sh
```

You will be prompted for:

- **PDF path** – for example `data/Annual_Report_JUNE-2017.pdf`.
- **Optional question** – for full QA mode.
- **Mode selection**:
  - `1) Triage only` – run just the Triage Agent and emit a `DocumentProfile`.
  - `2) Extraction` – run the multi-strategy extractor, update the extraction ledger, and cache an `ExtractedDocument` under `.refinery/extracted/`.
  - `3) Full QA` – end-to-end pipeline (triage → extraction → chunking → PageIndex → QA) with optional LDU preview from the vector DB.
  - `4) Chunk only` – build LDUs and ingest into the vector store from an existing cached extraction (no re-extraction).
  - `5) Vector DB` – explore stored chunks (LDUs) via a small CLI explorer (summary, list docs, show doc, search, raw dump).

All modes load configuration from `rubric/extraction_rules.yaml` and update artifacts in `.refinery/`.

### LLM Configuration via `.env`

LLM and VLM backends are configured **via environment variables**, which can be placed in a `.env` file at the project root. The code automatically loads `.env` using `python-dotenv`.

Example `.env` for OpenRouter-style backend:

```env
LLM_BACKEND=openrouter
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_API_KEY=sk-your-key
LLM_TEXT_MODEL=gpt-4o-mini
LLM_VISION_MODEL=gpt-4o-mini
```

Example `.env` for Ollama:

```env
LLM_BACKEND=ollama
LLM_API_BASE=http://localhost:11434
LLM_TEXT_MODEL=llama3.1:8b
LLM_VISION_MODEL=llama3.2-vision
```

You can switch providers or models by editing `.env` without touching the code.

### Programmatic Full Pipeline Demo

You can also call the pipeline directly from Python (e.g., for integration tests or notebooks):

```bash
python scripts/run_pipeline.py data/your_doc.pdf \
  --question "What are the main findings of this report?" \
  --show-ldu-preview
```

This will:

- Print the `DocumentProfile`.
- Run the multi-strategy extractor with escalation and log to `.refinery/extraction_ledger.jsonl`.
- Build LDUs and a basic `PageIndex`, ingest chunks into the vector store, and populate the `FactTable`.
- Ask the query agent your question and output an answer plus a `ProvenanceChain` (document, page, bbox, content_hash, snippet).
- Optionally preview a few stored LDUs as they appear in the vector DB.

### Notes

- **Domain onboarding notes** live in `DOMAIN_NOTES.md` (Phase 0).
- This project is intentionally modular so that:
  - New document types can be onboarded via `extraction_rules.yaml`.
  - Strategies can be swapped (e.g., different VLM provider) without touching the pipeline core.

