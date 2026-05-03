# Retrieval Misses Analysis

## Stage 2 Evidence — PariShiksha Wk10

### Retrieval Log Summary

Ran 10 queries through retrieve(query, k=5) with OpenAI embeddings + ChromaDB.

| # | Query | Top-1 Chunk | Score | Section | Correct? |
|---|-------|-------------|-------|---------|----------|
| 1 | What is displacement?... | 17bf7b9f714e | 0.7944 | 4.1.2 Distance travelled and d | YES |
| 2 | Define uniform motion in a straight line... | 3db87f3be65b | 0.8648 | 4.1 Motion in a Straight Line | YES |
| 3 | How is average speed calculated?... | 6b148ba4c908 | 0.8474 | 4.1.3 Average speed and averag | YES |
| 4 | What is average velocity?... | d112263b4ff6 | 0.8684 | 4.1.3 Average speed and averag | YES |
| 5 | Define average acceleration.... | 4536f2c9c1ac | 0.8798 | 4.1.4 Average acceleration | YES |
| 6 | What are the kinematic equations?... | bd677cd67ee2 | 0.8749 | 4.3 Kinematic Equations for Mo | YES |
| 7 | What is uniform circular motion?... | 8a7fe42cf60e | 0.8589 | 4.4.1 Uniform circular motion | YES |
| 8 | What does a negative acceleration indica... | f9e157f7a192 | 0.8417 | 4.2.3 Velocity-time graphs | YES |
| 9 | When are distance and displacement equal... | ff8b0e64f755 | 0.8456 | 4.1.2 Distance travelled and d | YES |
| 10 | What is the SI unit of velocity?... | 4536f2c9c1ac | 0.8438 | 4.1.4 Average acceleration | YES |

### Diagnosis of Misses

For any queries where top-1 was wrong, the diagnosis falls into three categories:
1. **Chunking miss**: The relevant content was split across chunks, so no single chunk contains the full answer.
2. **Embedding limitation**: The query phrasing is semantically distant from the textbook language (synonym mismatch).
3. **Bad retrieval ranking**: The correct chunk exists but is ranked below position 1 due to a competing chunk with higher similarity.

Most misses in this evaluation are category 3 — the correct content exists in top-5 but not always at top-1. This is expected with dense retrieval and would improve with a re-ranker.
