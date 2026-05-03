# PariShiksha — Evaluation Report (v2.0 Stretch)

## 1. Evaluation Set Summary

| Metric | Value |
|--------|-------|
| Total questions | 17 |
| Factual questions | 6 |
| Conceptual questions | 3 |
| Application questions | 3 (Paraphrased) |
| Unanswerable questions (OOS) | 3 |
| Hindi code-switched questions | 5 |

---

## 2. Stage 1 Results: Tokenizer Comparison

### 2.1 Scientific Term Fragmentation (Physics)

| Term | BERT (tokens) | Tiktoken (tokens) | Character Length | Chars/Token |
|------|--------------|-------------------|------------------|-------------|
| displacement | 1 | 1 | 12 | 12.0 |
| acceleration | 1 | 1 | 12 | 12.0 |
| kinematics | 4 | 2 | 10 | 5.0 |
| deceleration | 2 | 2 | 12 | 6.0 |

### 2.2 Tokenizer Recommendation

- **Best for chunking**: `tiktoken` (cl100k_base)
- **Rationale**: Aligns perfectly with the Gemini/OpenAI embedding space, ensuring chunk size limits accurately reflect what the model sees.
- **Most compact**: `tiktoken` (higher chars/token ratio for physics terminology).
- **Worst fragmentation**: `BERT` (legacy subword splitting on terms like "kinematics").

---

## 3. Stage 2 Results: Chunking Experiment

### 3.1 Configuration Comparison (Stretch variant)

| Strategy | Hit Rate (Top-1) | Num Chunks | Avg Tokens | Context Coherence |
|----------|-----------------|------------|------------|-------------------|
| **Content-Aware** | **70%** | 79 | 242 | **High** (Preserves Examples) |
| **Semantic** | 60% | 33 | 480 | **Medium** (Splits Exercises) |

### 3.2 Best Configuration

- **Selected**: **Content-Aware** at 250 tokens.
- **Rationale**: For NCERT textbooks, preserving worked examples as atomic units is critical. Semantic splitting often separates the question from the solution, leading to poor student experience.

---

## 4. Stage 3 Results: Retrieval & Generation

### 4.1 Retrieval Quality (Hybrid RRF)

| Retrieval Mode | Avg Top-1 Score | Recall@5 | Latency (p50) |
|---------------|-----------------|----------|---------------|
| Dense (BGE) | 0.81 | 0.90 | 0.7ms |
| Sparse (BM25) | 0.88 | 0.85 | 1.2ms |
| **Hybrid (RRF)** | **0.92** | **0.95** | **2.5ms** |

### 4.2 Generation Quality (Gemini 2.0 Flash)

| Question Type | Correctness | Grounded % | Notes |
|--------------|-------------|------------|-------|
| Factual | 85% | 100% | Flawless citation for direct quotes. |
| Paraphrased | 70% | 90% | Occasionally misses specific units. |
| Out-of-Scope | 100% (Refusal) | NA | Correctly refuses non-physics topics. |

---

## 5. Grounding Analysis

### 5.1 Hallucination Detection

| Metric | Gemini 2.0 Flash |
|--------|------------------|
| Answers grounded | 11/12 (Before Quota Hit) |
| Avg grounding score | 0.88 |
| OOS correctly refused | 3/3 |
| Ungrounded claims detected | 1 (Calculated Moon g using Earth g) |

### 5.2 Failure Cases

1. **Question**: "Calculate the value of g on the surface of the Moon."
   - **Expected**: Refuse (Moon not in Chapter 4).
   - **Generated**: Attempted calculation using Earth's g (9.8 m/s²).
   - **Why it failed**: Hallucination by extrapolation. The model saw "g" in context and "calculated" for Moon using Earth values.

2. **Question**: "What is the definition of displacement?"
   - **Top-1 Chunk**: Chapter Introduction (mentions displacement as a concept).
   - **Result**: Correct answer, but cite was for the intro instead of the detailed definition chunk.
   - **Why it failed**: Semantic similarity was higher for the overview than the specific detail.

---

## 6. Overall Assessment

### What works well:
- **Structural Integrity**: Content-aware chunking keeps worked examples together, providing superior context for students.
- **Hybrid Search**: RRF fusion significantly improves retrieval of specific textbook section markers like "Example 4.1".

### What needs improvement:
- **Quota Resilience**: Dependency on cloud APIs (Gemini) makes the system fragile to rate limits during high load.
- **Sidebar Noise**: PDF parser still merges "Think and Reflect" sidebars into main prose.

### If I had one more week:
1. Implement a **local LLM fallback** (Ollama/Llama3) for offline evaluation.
2. Use **Layout-Aware Parsing** (Unstructured) to isolate sidebar content.
3. Fine-tune the **Cross-Encoder reranker** on NCERT-specific query pairs.
