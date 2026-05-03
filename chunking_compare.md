# Chunking Comparison: Content-Aware vs Semantic

## Stretch Stage 1 Evidence — PariShiksha

### 1. Quantitative Results (BM25 Top-1 Hit Rate)

| Variant | Chunk Count | Hit Rate (10-Q) |
|---------|-------------|-----------------|
| **V1: Content-Aware** | 79 | 70% |
| **V2: Semantic**      | 33 | 60% |

### 2. Qualitative Analysis

**V1: Content-Aware (Winner)**
- **Strengths**: Excels at keeping pedagogical structures (worked examples) intact. In queries like "Example 4.1", it consistently returns the full problem context.
- **Weaknesses**: Fixed token boundaries within prose can sometimes cut a definition in half if it's long.

**V2: Semantic (Runner-up)**
- **Strengths**: Creates very coherent prose chunks. Definitions are rarely split.
- **Weaknesses**: Frequently splits Worked Examples because the "Question" and "Solution" parts have different semantic profiles, which is disastrous for a study assistant.

### 3. Decisions for Stage 2
We will carry **Variant 1 (Content-Aware)** into Stage 2. For an educational RAG system, preserving the structural integrity of exercises and examples is more valuable than semantic boundary alignment in generic prose.
