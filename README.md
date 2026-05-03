# PariShiksha — NCERT Science Study Assistant v2.0 (STRETCH TRACK)

> **"Bridging the Classroom Gap with Truth-Bound AI"**
> Week 10 · Stretch Track · PG Diploma in AI-ML & Agentic AI Engineering · Cohort 2

PariShiksha is a production-ready, NCERT-grounded study assistant for Class 9 Science
(Chapter 4: Describing Motion Around Us). It implements a **content-type-aware RAG pipeline**
with strict grounding, honest evaluation, and citation-enforced generation.

**NCERT Source:** [https://ncert.nic.in/textbook.php?iesc1=0-11](https://ncert.nic.in/textbook.php?iesc1=0-11)

---

## 🏗️ Architecture (v2.0)

```
┌─────────────────────────────────────────────────────────────┐
│                    Student Question                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL: Google text-embedding-004 + ChromaDB            │
│  - PersistentClient at ./chroma_wk10                        │
│  - Cosine similarity, top-k=5                               │
│  - Chunk metadata: {source, section, content_type, page}    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  GENERATION: Google Gemini 1.5 Flash @ temperature=0        │
│  - Strict prompt with [Source: chunk_id] citations          │
│  - Clean refusal: "I don't have that in my study materials" │
│  - Anti-extrapolation rules for plausibly-answerable OOS    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  OUTPUT: {answer, sources, chunk_ids}                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
parishiksha/
├── wk10_pipeline.py          # v2.0 pipeline orchestrator (all 5 stages)
├── wk10_chunker.py           # Stage 1: Content-type-aware chunking (tiktoken)
├── wk10_embedder.py          # Stage 2: Gemini embeddings + ChromaDB retrieval
├── wk10_ask.py               # Stage 3: ask() with Gemini 1.5 Flash + strict prompt
├── wk10_eval.py              # Stage 4+5: 12-Q evaluation + targeted fix
├── wk10_chunks.json          # Persisted chunks with content_type metadata
├── chroma_wk10/              # ChromaDB persistent storage
│
├── main.py                   # Wk9 legacy pipeline (preserved)
├── config/config.py          # Central configuration
├── src/                      # Wk9 modules (extraction, chunking, retrieval, etc.)
├── data/                     # Raw PDFs + processed text
│   ├── raw/                  # NCERT PDFs (not committed)
│   └── processed/            # Cleaned text + sections JSON
│
├── reflection.md             # Wk10 reflection questionnaire
├── chunking_diff.md          # Stage 1 evidence: Wk9 vs Wk10 comparison
├── retrieval_log.json        # Stage 2 evidence: top-1 results for 10 queries
├── retrieval_misses.md       # Stage 2 evidence: miss diagnosis
├── prompt_diff.md            # Stage 3 evidence: permissive vs strict prompt
├── eval_raw.csv              # Stage 4 evidence: raw ask() output
├── eval_scored.csv           # Stage 4 evidence: hand-scored 3-axis
├── eval_v2_scored.csv        # Stage 5 evidence: post-fix scores
├── fix_memo.md               # Stage 5 evidence: fix description + delta
│
├── requirements.txt          # Python dependencies
├── .env.example              # API key placeholders
└── .gitignore
```

---

## 🚀 Quick Start (Fresh Clone)

### 1. Install Dependencies

```bash
git clone <repository-url>
cd parishiksha
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add:
#   GEMINI_API_KEY=AIzaSy...
```

### 3. Run the Wk10 Pipeline

```bash
# Run all 5 stages
python wk10_pipeline.py --stage all

# Or run individual stages
python wk10_pipeline.py --stage chunk      # Stage 1: Chunking
python wk10_pipeline.py --stage embed      # Stage 2: Embedding + Retrieval
python wk10_pipeline.py --stage generate   # Stage 3: Generation + Prompt Comparison
python wk10_pipeline.py --stage evaluate   # Stage 4+5: Evaluation + Fix
```

### 4. Interactive Q&A

```bash
python wk10_ask.py
# Type questions and get grounded, cited answers
```

---

## 📊 Evaluation Summary (Core Track)

| Metric | v1 (Before Fix) | v2 (After Fix) |
|--------|-----------------|----------------|
| Correct (Y) | See eval_scored.csv | See eval_v2_scored.csv |
| Grounded (Y) | See eval_scored.csv | See eval_v2_scored.csv |
| OOS Refused | See eval_scored.csv | See eval_v2_scored.csv |

**Eval set:** 12 questions (6 direct + 3 paraphrased + 3 OOS including 1 plausibly-answerable).
**Scoring axes:** (a) correct Y/N/partial, (b) grounded Y/N, (c) refused_when_oos Y/N/NA.

---

## 📦 Wk10 Deliverables Checklist

| # | Deliverable | File |
|---|-------------|------|
| 1 | Chunks with content_type metadata | `wk10_chunks.json` |
| 2 | Chunking diff (Wk9 → Wk10) | `chunking_diff.md` |
| 3 | Retrieval log (10 queries) | `retrieval_log.json` |
| 4 | Retrieval miss diagnosis | `retrieval_misses.md` |
| 5 | ask() function | `wk10_ask.py` |
| 6 | Prompt comparison | `prompt_diff.md` |
| 7 | Raw eval output | `eval_raw.csv` |
| 8 | Hand-scored eval | `eval_scored.csv` |
| 9 | Post-fix eval | `eval_v2_scored.csv` |
| 10 | Fix memo | `fix_memo.md` |
| 11 | Reflection | `reflection.md` |

---

## 🔧 Technology Stack

| Component | Technology |
|-----------|-----------|
| Embedding | Google text-embedding-004 |
| Vector DB | ChromaDB (PersistentClient) |
| Generation | Google Gemini 1.5 Flash |
| Token counting | tiktoken (cl100k_base) |
| Chunking | Content-type-aware (prose/worked_example/exercise) |

---

## 📜 Wk9 → Wk10 Migration

- Wk9 final commit tagged as `v1.0-wk9`
- Wk10 work on `main` with feature branches `feat/v2-*`
- Final submission tagged as `v2.0-wk10`
- Wk9 code preserved in `src/` and `main.py` (not deleted)
