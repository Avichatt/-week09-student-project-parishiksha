# DB & Embedding Comparison

## Stretch Stage 2 Evidence — PariShiksha

### 1. Benchmark Results

| Combination | p50 Latency (ms) | p95 Latency (ms) | Recall@5 |
|-------------|------------------|------------------|----------|
| Chroma + BGE | 2.06 | 2.36 | 0.9 |
| Qdrant + BGE | 0.68 | 0.96 | 0.9 |
| Chroma + Gemini | 1.7 | 2.28 | 0.85 |

### 2. Analysis

**Winner: Qdrant + BGE**
- **Latency**: Local BGE embeddings remove the network overhead of cloud APIs, reducing p50 significantly. Qdrant (even in-memory) shows slightly better query execution speed than Chroma in this small-scale test.
- **Recall**: Gemini (3072-dim) technically captures more nuance, but for basic physics queries, the local BGE (384-dim) model is surprisingly competitive.

### 3. Scaling to 10× (1,000+ chunks)
1. **Write Throughput**: Qdrant's async batching and HNSW indexing become critical as collection size grows. Chroma's disk-based persistence starts showing overhead during large updates.
2. **Query Latency**: Network round-trips for Gemini embeddings will dominate search time. Moving to a local embedding service or using a faster cloud model (like OpenAI's small) would be necessary for concurrent users.
3. **Cost**: At scale, Gemini cost ($0.02 per 1M tokens) is negligible, but the 429 quota limits on free tiers make local models (BGE) the more reliable choice for development.
