## Domain Notes – Document Intelligence Refinery

This file captures **Phase 0** domain onboarding notes: extraction strategy decision tree, observed failure modes, and a high-level pipeline diagram. It is meant to evolve as you test against the corpus.

### Extraction Strategy Decision Tree (Initial Draft)

1. **Triage: origin type & complexity**
   - If `avg_char_density` is **high** (e.g. \> 0.002 chars / pt²) **and** `avg_image_area_ratio` \< 0.3  
     → classify as **`origin_type = native_digital`**.
   - If `avg_char_density` is **very low** (e.g. \< 0.0003 chars / pt²) **and** `avg_image_area_ratio` \>= 0.4  
     → classify as **`origin_type = scanned_image`**.
   - Otherwise → **`origin_type = mixed`**.

2. **Layout complexity (heuristic)**
   - Compute the distribution of text-block `x0` positions across pages.
   - If there are **two or more stable x-bands** with similar char density → **`multi_column`**.
   - If the proportion of table-like structures (ruled lines / aligned blocks) is high → **`table_heavy`**.
   - If image/figure count is high relative to text → **`figure_heavy`**.
   - Otherwise → **`single_column`** or **`mixed`**.

3. **Strategy selection**
   - If `origin_type = native_digital` **and** `layout_complexity = single_column`:
     - `estimated_extraction_cost = fast_text_sufficient`
     - **Use Strategy A – FastTextExtractor (pdfplumber)**.
   - Else if `layout_complexity in {multi_column, table_heavy, mixed}` **and** `origin_type != scanned_image`:
     - `estimated_extraction_cost = needs_layout_model`
     - **Use Strategy B – LayoutExtractor (Docling)**.
   - Else (`origin_type = scanned_image` or confidence low):
     - `estimated_extraction_cost = needs_vision_model`
     - **Use Strategy C – VisionExtractor (VLM)**.

4. **Escalation guard**
   - After Strategy A:
     - If **page coverage** \< 0.9 (too many empty/low-density pages) **or**
       average **page confidence** \< 0.6  
       → **escalate to Strategy B**.
   - After Strategy B:
     - If table detection fails (no tables but table-heavy layout expected) **or**
       average **page confidence** \< 0.6  
       → **escalate to Strategy C** (vision).

All thresholds above are externalized in `rubric/extraction_rules.yaml` and are expected to be refined empirically.

### Observed / Expected Failure Modes (To Refine With Corpus)

- **Structure collapse**:
  - Multi-column financial reports flattened into a single text stream.
  - Table headers separated from their rows, especially when spanning multiple pages.
- **Context poverty**:
  - Figure captions detached from figures.
  - Numbered lists split in the middle due to naive token-based chunking.
- **Provenance blindness**:
  - Extracted facts without page or bbox coordinates.
  - Inability to answer “where did this value come from?” reliably.
- **Scanned documents**:
  - OCR errors for small fonts or faint text.
  - Non-Latin scripts mis-recognized by generic OCR/VLM prompts.

These failure modes should be updated as you run the refinery on each of the four document classes.

### Pipeline Diagram (Mermaid)

```mermaid
flowchart TD
    A[Input Document] --> B[Triage Agent<br/>DocumentProfile]
    B --> C{Estimated Extraction Cost}
    C -->|fast_text_sufficient| D[Strategy A<br/>FastTextExtractor (pdfplumber)]
    C -->|needs_layout_model| E[Strategy B<br/>LayoutExtractor (Docling)]
    C -->|needs_vision_model| F[Strategy C<br/>VisionExtractor (VLM)]

    D --> G{Confidence OK?}
    G -->|no| E
    G -->|yes| H[ExtractedDocument]

    E --> I{Confidence OK?}
    I -->|no| F
    I -->|yes| H

    F --> H

    H --> J[Semantic Chunking Engine<br/>LDU list]
    J --> K[PageIndex Builder<br/>Section Tree]
    J --> L[Vector Store<br/>LDU embeddings]
    J --> M[FactTable Extractor<br/>SQLite]

    K --> N[Query Agent<br/>LangGraph]
    L --> N
    M --> N

    N --> O[Answer + ProvenanceChain]
```

This diagram should stay in sync with the actual implementation in `src/agents/`.

