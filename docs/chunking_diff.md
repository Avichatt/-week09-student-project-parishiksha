# Chunking Diff: Wk9 → Wk10

## Stage 1 Evidence — PariShiksha

### What changed

**Wk9 chunking** used a BERT-based tokenizer (`bert-base-uncased`) with three strategies (fixed-token, sentence-based, semantic-paragraph) at sizes [128, 256, 512]. Chunks had no content-type metadata — all chunks were treated identically regardless of whether they contained prose, worked examples, or exercises. Worked examples and tables were frequently split across chunk boundaries, losing the question-answer structure that is critical for a study assistant.

**Wk10 chunking** switches to `tiktoken` (cl100k_base) for token counting, which aligns with the OpenAI embedding model. Each chunk now carries `content_type` metadata: `prose`, `worked_example`, or `question_or_exercise`. Worked examples (Example 4.1, 4.2, etc.) are preserved as complete units. Target size is ~250 tokens. Section headings are injected as prefixes for retrieval context.

### Wk10 chunk statistics
- Total chunks: 79
- Avg tokens: 206
- Min/Max: 31/325
- Content type distribution:
  - `prose`: 39 chunks
  - `question_or_exercise`: 22 chunks
  - `worked_example`: 18 chunks

### Where content_type filtering would have helped

In Wk9, retrieval for "Solve Example 4.3" would return prose chunks about acceleration alongside fragments of the worked example. With Wk10's `content_type: worked_example` metadata, we can filter to retrieve only complete worked examples, giving the student the full question + solution.
