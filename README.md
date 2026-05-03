# PariShiksha — NCERT Science Study Assistant v2.0 (STRETCH TRACK)

> **"Bridging the Classroom Gap with Truth-Bound AI"**
> Week 10 · Stretch Track · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 2

PariShiksha is a production-ready, NCERT-grounded study assistant for Class 9 Science
(Chapter 4: Describing Motion Around Us). It implements a **content-type-aware RAG pipeline**
with strict grounding, honest evaluation, and citation-enforced generation.

**NCERT Source:** [https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

---

## 🏗️ Architecture (v2.0 Stretch)

```mermaid
graph TD
    User([Student Question]) --> MQ[MultiQuery: Gemini Rewrites x3]
    
    subgraph Retrieval
        MQ --> BM25[BM25 Retrieval]
        MQ --> Dense[Dense: gemini-embedding-001 + Chroma/Qdrant]
        BM25 --> RRF[RRF Fusion]
        Dense --> RRF
    end
    
    subgraph Post-Processing
        RRF --> Rerank[Local Cross-Encoder Reranker]
    end
    
    subgraph Generation
        Rerank --> Ask[Gemini 2.0 Flash + Strict Prompt]
    end
    
    Ask --> Output{Grounded Answer + Citations}
```

---

## 📽️ Submission Video
**Loom Link:** [https://www.loom.com/share/placeholder](https://www.loom.com/share/placeholder) (5 min Stretch Walkthrough)

---

## 📂 Project Structure (v2.0)

```
parishiksha/
├── src/                      # Core Logic & Pipeline Stages
│   ├── pipeline.py           # Core orchestrator
│   ├── stretch_pipeline.py   # Stretch track orchestrator
│   ├── chunking.py           # Content-type-aware splitter
│   ├── retrieval.py          # Embedding + Vector search
│   ├── generation.py         # Gemini-powered generation
│   ├── evaluation.py         # Automated & manual scoring
│   └── stretch_s1..s4.py     # Stretch track logic modules
│
├── data/                     # Data persistence
│   ├── processed/            # Cleaned NCERT sections
│   └── results/              # Chunks, CSVs, and JSON logs
│
├── docs/                     # Submission Artifacts
│   ├── reflection.md         # Final questionnaire
│   ├── failure_memo.md       # Architecture & Failure analysis
│   ├── chunking_compare.md   # Chunker variant study
│   └── *_diff.md             # Evidence documents
│
├── wk10_pipeline.py          # Root entry point (Core)
├── wk10_stretch_pipeline.py  # Root entry point (Stretch)
├── requirements.txt          # Pinned dependencies
└── .gitignore
```

---

## 🚀 Quick Start (Stretch Track)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Stretch Pipeline
```bash
# This will run comparison, benchmarking, hybrid retrieval, and reranking
python wk10_stretch_pipeline.py
```

---

## 📊 Evaluation Summary

- **Faithfulness Target**: ≥ 0.7 (RAGAS)
- **Hybrid Retrieval**: BM25 + Dense (Fused via RRF)
- **Reranking**: Local `ms-marco-MiniLM-L-6-v2`
- **MultiQuery**: Gemini-powered query expansion (x3)

---

## 📦 Wk10 Deliverables Checklist

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Chunks with metadata | `data/results/wk10_chunks.json` |
| 2 | Chunking comparison | `docs/chunking_compare.md` |
| 3 | Retrieval log | `data/results/retrieval_log.json` |
| 4 | DB Benchmark | `data/results/db_benchmark.csv` |
| 5 | RAGAS Report | `data/results/ragas_report.csv` |
| 6 | Fix Memo | `docs/fix_memo.md` |
| 7 | Reflection | `docs/reflection.md` |

---

## 🔧 Technology Stack (v2.0)

| Component | Technology |
|-----------|-----------|
| Embedding | Google gemini-embedding-001 |
| Vector DB | ChromaDB + Qdrant (Local) |
| Generation | Google Gemini 2.0 Flash |
| Reranker | Local CrossEncoder (ms-marco-MiniLM) |
| MultiQuery | Gemini 2.0 Flash |
| Token counting | tiktoken (cl100k_base) |
| Chunking | Content-type-aware + Semantic Chunker |

---

## 📜 Wk9 → Wk10 Migration

- Wk9 final commit tagged as `v1.0-wk9`
- Wk10 work on `main` with feature branches `feat/v2-*`
- Final submission tagged as `v2.0-wk10`
- Wk9 code preserved in `src/` and `main.py` (not deleted)
