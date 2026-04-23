# PariShiksha — System Architecture

## Overview

PariShiksha is a retrieval-augmented generation (RAG) system for NCERT Science content. This document describes the system architecture, data flow, and key technical decisions.

---

## High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐
│  NCERT PDF  │────▶│  Extraction  │────▶│   Chunking    │────▶│  Embedding  │
│  (raw)      │     │  + Cleaning  │     │  (3 strategies)│    │  (SBERT+TF) │
└─────────────┘     └──────────────┘     └───────────────┘     └──────┬──────┘
                                                                       │
                    ┌──────────────┐     ┌───────────────┐            │
                    │  Evaluation  │◀────│  Generation   │◀───────────┘
                    │  (20+ Qs)    │     │ (Gemini / T5) │    Hybrid Retrieval
                    └──────────────┘     └───────────────┘
```

---

## Data Flow

### Stage 1: Extraction Pipeline

```
PDF File (NCERT)
    │
    ├── PyMuPDF (fitz) ──────┐
    │   - Fast extraction     │
    │   - Text blocks + bbox  ├──▶ Quality Cross-Validation
    │   - Image detection     │        (character count comparison)
    │                         │
    └── pdfplumber ──────────┘
        - Table extraction           │
        - Word-level positions       ▼
                                 
                            Cleaning Pipeline
                            ├── Header/footer removal
                            ├── Mojibake fixing
                            ├── Whitespace normalization
                            ├── Page-break hyphenation repair
                            ├── Section boundary detection
                            └── Content type classification
                                     │
                                     ▼
                            Structured Output
                            ├── chapter_X_structured.json
                            ├── chapter_X_clean.txt
                            └── chapter_X_sections.json
```

### Stage 2: Chunking Strategies

```
Clean Text + Sections
    │
    ├── Strategy 1: Fixed-Token
    │   - Deterministic token windows
    │   - Configurable overlap (15%)
    │   - Risk: splits mid-sentence
    │
    ├── Strategy 2: Sentence-Based
    │   - Groups N sentences to target size
    │   - Preserves sentence boundaries
    │   - Risk: uneven chunk sizes
    │
    └── Strategy 3: Semantic-Paragraph
        - Respects section headings
        - Section header prepended to each chunk
        - Risk: very uneven sizes
                │
                ▼
        Each strategy × [128, 256, 512] tokens
        = 9 chunk configurations per chapter
```

### Stage 3: Retrieval & Generation

```
Student Question
    │
    ├── Dense Path (SBERT: all-MiniLM-L6-v2)
    │   - Encodes query as 384-dim vector
    │   - Cosine similarity against chunk embeddings
    │   - Handles paraphrased queries
    │
    └── Sparse Path (TF-IDF)
        - Transforms query with fitted vectorizer
        - Cosine similarity on sparse vectors
        - Handles exact keyword matching
                │
                ▼
        Hybrid Score = α × dense + (1-α) × sparse
        (α = 0.7 default, tunable)
                │
                ▼
        Top-K Chunks (K=5)
                │
                ├── Gemini (decoder-only)
                │   - Full context window
                │   - Grounding system prompt
                │   - Temperature 0.3
                │
                └── Flan-T5 (encoder-decoder)
                    - Encoder: bidirectional context
                    - Decoder: left-to-right generation
                    - Beam search (n=4)
                            │
                            ▼
                    Grounding Verification
                    ├── Lexical overlap (40% weight)
                    ├── Sentence-level scoring (60% weight)
                    └── Refusal detection
```

---

## Component Responsibilities

| Component | Responsibility | Key Design Decision |
|-----------|---------------|-------------------|
| `PDFExtractor` | Raw text extraction | Dual-backend for quality cross-validation |
| `TextCleaner` | Cleaning & structuring | NCERT-specific regex patterns, content type detection |
| `TokenizerAnalyzer` | Tokenizer comparison | Measures fragmentation of scientific vocabulary |
| `TextChunker` | Text segmentation | Three strategies × multiple sizes for systematic comparison |
| `ChunkEmbedder` | Vector representation | Both dense (SBERT) and sparse (TF-IDF) for hybrid search |
| `HybridRetriever` | Context retrieval | α-weighted combination of dense and sparse scores |
| `AnswerGenerator` | Answer generation | Dual architecture: Gemini (decoder-only) + T5 (encoder-decoder) |
| `GroundingChecker` | Hallucination detection | Multi-level verification: lexical → sentence → refusal |
| `EvalSetBuilder` | Test set management | 5 question types including unanswerable and code-switched |
| `PariShikshaEvaluator` | End-to-end evaluation | Full pipeline scoring with per-type aggregation |

---

## Configuration Architecture

All configuration is centralized in `config/config.py`:

- **Paths**: All directory paths derived from `PROJECT_ROOT`
- **Models**: Tokenizer names, embedding models, generation models
- **Hyperparameters**: Chunk sizes, overlap ratios, temperature, top-k
- **Evaluation**: Question types, minimum question counts, metrics
- **API Keys**: Loaded from `.env` file via `python-dotenv`

This ensures zero magic strings scattered across the codebase.

---

## Error Handling & Graceful Degradation

1. **Missing PDFs**: Auto-download attempted, clear error message with manual download URL
2. **API failures**: Gemini errors caught, logged, and returned as error objects (not crashes)
3. **Missing T5 model**: T5 evaluation gracefully skipped if model can't be loaded
4. **No retrieval index**: Evaluation can run in generation-only mode
5. **Empty chunks**: Filtered by minimum token threshold (50 tokens)
