## Document Intelligence Refinery

This repository implements **The Document Intelligence Refinery** – a production-inspired, multi-stage agentic pipeline for converting heterogeneous enterprise documents into structured, queryable, spatially-indexed knowledge.

### Features (High Level)

- **Triage Agent (`src/agents/triage.py`)**: Profiles each document (origin type, layout complexity, language, domain hint, estimated extraction cost) and writes `DocumentProfile` JSON to `.refinery/profiles/`.
- **Multi-Strategy Extraction (`src/agents/extractor.py`, `src/strategies/`)**:
  - Fast text extraction (pdfplumber) with confidence scoring.
  - Layout-aware extraction (Docling) for complex/table-heavy documents.
  - Vision-augmented extraction (VLM via HTTP API) for scanned/low-confidence pages.
  - Escalation guard and extraction ledger in `.refinery/extraction_ledger.jsonl`.
- **Semantic Chunking & PageIndex (Phase 3)**:
  - Normalized `ExtractedDocument` model and `LDU` chunks.
  - PageIndex tree to navigate long documents before vector search.
- **Query Agent (Phase 4)**:
  - LangGraph-based agent with `pageindex_navigate`, `semantic_search`, and `structured_query` tools.
  - Every answer includes a provenance chain (document, page, bbox, content_hash).

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
- `rubric/extraction_rules.yaml` – Externalized thresholds and chunking rules.
- `.refinery/` – Runtime artifacts:
  - `profiles/` – `DocumentProfile` JSON outputs.
  - `pageindex/` – PageIndex JSON trees.
  - `extraction_ledger.jsonl` – extraction trace and cost estimates.
- `tests/` – Unit tests for triage and extraction confidence scoring.

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

### Running Triage & Extraction (CLI sketch)

Once implemented, a typical flow for a PDF at `data/sample.pdf` will look like:

```bash
python -m src.agents.triage data/sample.pdf
python -m src.agents.extractor data/sample.pdf
```

Each command will:

- Load configuration from `rubric/extraction_rules.yaml`.
- Produce/update artifacts in `.refinery/`.

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

### Full Pipeline Demo

To run the **end-to-end pipeline** (Triage → Extraction → Chunking → PageIndex → Query with provenance) on a single PDF:

```bash
python scripts/run_pipeline.py data/your_doc.pdf \
  --question "What are the main findings of this report?"
```

This will:

- Print the `DocumentProfile`.
- Run the multi-strategy extractor with escalation and log to `.refinery/extraction_ledger.jsonl`.
- Build LDUs and a basic `PageIndex`.
- Ask the query agent your question and output an answer plus a `ProvenanceChain` (document, page, bbox, content_hash, snippet).

### Notes

- **Domain onboarding notes** live in `DOMAIN_NOTES.md` (Phase 0).
- This project is intentionally modular so that:
  - New document types can be onboarded via `extraction_rules.yaml`.
  - Strategies can be swapped (e.g., different VLM provider) without touching the pipeline core.

