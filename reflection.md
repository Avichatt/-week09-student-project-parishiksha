# Reflection Questionnaire — PariShiksha v2.0 (Wk10)

## 1. What was your chunking strategy in Wk10 and why did you change it from Wk9?

In Wk9, I used three chunking strategies (fixed-token, sentence-based, semantic-paragraph) with
a BERT tokenizer (`bert-base-uncased`) at sizes [128, 256, 512]. The chunks had no content-type
metadata — everything was treated as generic text.

In Wk10, I switched to **content-type-aware chunking** with `tiktoken` (cl100k_base) at a target
of ~250 tokens. Each chunk is classified as `prose`, `worked_example`, or `question_or_exercise`
using regex detection on markers like `Example 4.1:`, `Activity 4.2:`, and `Pause and Ponder`.

**Why the change:** Worked examples (e.g., Example 4.3 — the bus acceleration problem) were being
split across multiple chunks in Wk9, losing the question-answer structure. A student asking
"Solve the bus braking problem" would get a fragment of the solution. With Wk10's content-type
preservation, the entire worked example (question + given values + solution steps + answer) stays
in one chunk. The `content_type` metadata also enables future filtering — retrieve only worked
examples when the query looks like a homework problem.

**Real chunk_ids affected:** Chunks with `content_type: worked_example` in `wk10_chunks.json`
(e.g., chunks covering Example 4.1 through Example 4.8) are now preserved as complete units
instead of being split at arbitrary token boundaries.

## 2. How does your retrieval pipeline work and what are its honest limitations?

The pipeline uses **OpenAI text-embedding-3-small** (1536-dim) for dense embeddings, persisted
in **ChromaDB** (PersistentClient at `./chroma_wk10`) with cosine similarity. Retrieval is
top-k=5 dense search with no sparse component (Core track).

**Honest limitations:**
- No hybrid retrieval: Pure dense search misses exact keyword matches. A query for "9.8 m/s²"
  won't necessarily surface the chunk containing that exact number.
- No re-ranker: The top-1 result from embedding similarity isn't always the best answer. In
  `retrieval_log.json`, some queries have the correct chunk at position 2-3 rather than 1.
- Single embedding model: text-embedding-3-small is good but not optimized for scientific text.
  Terms like "retardation" (NCERT's word for deceleration) may not embed close to "slowing down."

## 3. How does your grounding/refusal mechanism work?

The system uses **Anthropic claude-haiku-4-5** at temperature=0 with a strict prompt that:
1. Requires citing `[Source: chunk_id]` after every factual claim
2. Forces exact refusal: "I don't have that in my study materials."
3. Prohibits calculation/inference beyond what's explicitly stated

The v2 prompt (Stage 5 fix) adds explicit OOS examples to help the model recognize
"plausibly-answerable" OOS queries — e.g., "Calculate g on the Moon" where the formula exists
in the context but Moon-specific values don't.

**Real eval evidence:** In `eval_scored.csv`, the OOS questions (OOS1, OOS2, OOS3) show whether
the refusal mechanism worked. The plausibly-answerable OOS3 ("Calculate g on the Moon") is the
hardest test — the model sees g=9.8 m/s² in the context and must resist calculating Moon gravity.

## 4. What was your worst evaluation failure and what did you learn from it?

The worst failure was **OOS3: "Calculate the value of g on the surface of the Moon."** This is a
"plausibly-answerable" OOS question — the formula for g and Earth's value (9.8 m/s²) are in
Chapter 4, but Moon-specific values are not. The permissive prompt hallucinated a Moon gravity
calculation. Even the initial strict prompt sometimes extrapolated.

**What I learned:** Strict grounding isn't just about "is the topic in the corpus?" — it's about
"is the *specific scenario* in the corpus?" A model that sees `g = 9.8 m/s²` and a question
about Moon gravity will naturally try to help by calculating. The fix required explicitly telling
the model: "Do NOT calculate values not explicitly given."

This maps to the **mixed structure** failure mode from the catalog — the retriever surfaces
relevant-looking content (kinematic equations, g value) but the generation step misinterprets
scope.

## 5. What is one industry technique you would explore in the next 6 months?

**CRAG (Corrective Retrieval Augmented Generation)** — Yan et al., 2024. CRAG adds a lightweight
evaluator between retrieval and generation that scores retrieved documents as "Correct,"
"Incorrect," or "Ambiguous." For "Incorrect" retrievals, it triggers a web search fallback. For
"Ambiguous" cases, it refines the query.

For PariShiksha, CRAG would directly address the plausibly-answerable OOS problem: the evaluator
would catch cases where retrieved chunks are topically related but don't actually contain the
answer, and route to a clean refusal instead of generation. This is more robust than prompt
engineering alone because it makes the decision *before* the LLM sees the context, reducing
hallucination risk at the architectural level.

## 6. What would you do differently if starting over?

1. **Start with evaluation, not chunking.** I should have written the 12-question eval set on
   Day 1 and used it to guide every design decision. Instead, I optimized chunking first and
   only discovered its weaknesses during evaluation.

2. **Use a simpler chunking strategy and iterate.** The Wk9 three-strategy experiment was
   interesting but unnecessary for a working system. One good content-type-aware chunker with
   measurement (Wk10's approach) is better than three unvalidated strategies.

3. **Track cost.** I didn't track API costs for OpenAI embeddings or Claude Haiku calls. In
   production, the embedding cost for re-indexing and per-query generation cost would be the
   first things I'd instrument.
