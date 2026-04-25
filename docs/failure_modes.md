# PariShiksha Failure Modes Analysis

This document identifies the top three production failure modes for the PariShiksha RAG pipeline, grounded in the results of the 20-question evaluation benchmark conducted on April 25, 2026.

## 1. Upstream Quota Exhaustion (HTTP 429)
**Observation:** During the batch evaluation of 20 questions, the Gemini model successfully processed only the first 6 questions before returning `429 You exceeded your current quota`.
**Impact:** In a production environment, this translates to intermittent service outages for students. Even if the RAG pipeline (retrieval) is functional, the generation layer becomes a bottleneck under high concurrency (e.g., a classroom of 30 students asking questions simultaneously).
**Grounded Evidence:** `evaluation_report_gemini.json` shows multiple failures with the error: `Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests`.
**Mitigation:** Implement exponential backoff in `AnswerGenerator`, utilize a pay-as-you-go tier with higher TPM limits, or provide a local fallback model (like Flan-T5) when the primary API is throttled.

## 2. Context Window Truncation and Generation Prematurity
**Observation:** Several factual answers were truncated mid-sentence. For example, the definition for "displacement" was cut off after the introductory phrase.
**Impact:** Students receive incomplete or confusing pedagogical information, which can lead to misconceptions about critical scientific definitions.
**Grounded Evidence:** Question ID `273cbaaf...` returned: `Based on the textbook context provided, here is the definition of displacement: **`.
**Probable Cause:** This failure mode occurs when the combined length of the system prompt and the 5 retrieved chunks approaches the model's output limit or when transient API jitter interrupts the response stream.
**Mitigation:** Optimize chunk size to ensure the top-3 chunks fit comfortably within the generation window, and implement a validation check in the `GuardrailVerifier` to reject and retry truncated outputs.

## 3. Keyword Mismatch and Evaluation False Negatives
**Observation:** The evaluation scores for "Conceptual" and "Paraphrased" questions were consistently lower (approx. 30-50%) than direct factual queries, even when the answers were semantically correct.
**Impact:** This failure mode reflects a disconnect between the **Evaluation Schema** and the **Model Creativity**. If a student understands the concept but the evaluator expects exact textbook keywords, the system's perceived "Accuracy" drops, leading to incorrect performance reporting to stakeholders.
**Grounded Evidence:** Question ID `c22e5587...` showed a low keyword coverage score despite the model correctly explaining non-uniform motion, because it used "changing velocity" instead of the exact textbook phrase "unequal distances in equal intervals of time."
**Mitigation:** Move towards LLM-based evaluation (using a "Teacher" model) to grade for semantic correctness rather than literal keyword matching, and expand the `must_include` lists in `EvalSetBuilder` to include common scientific synonyms.
