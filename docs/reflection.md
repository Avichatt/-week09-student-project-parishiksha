# PariShiksha — Reflection Questionnaire

> **Instructions**: Answer each question honestly. Effort and honesty matter more than polish.

---

## 1. Tokenizer Choice and Its Downstream Effects

**Question**: You compared at least two tokenizers on NCERT Science vocabulary. Which tokenizer would you choose for production, and why? What would change downstream (chunk sizes, retrieval quality, generation behavior) if you picked a different one?

**Your Answer:**

_[Write here]_

---

## 2. Chunking Strategy Trade-offs

**Question**: Describe the trade-off between chunk size and retrieval quality that you observed. What happens when chunks are too small? Too large? Which strategy worked best for NCERT content and why?

**Your Answer:**

_[Write here]_

---

## 3. Why Not GANs?

**Question**: This week you studied GANs alongside transformers. Explain why GAN-style adversarial generation is the wrong tool for a grounded textbook QA system like PariShiksha. What would go wrong if you tried to use a GAN-generated answer?

**Your Answer:**

_[Write here]_

### Suggested answer framework:
- GANs optimize for distributional similarity (generating outputs that look like real data), not for factual accuracy
- The discriminator cannot tell if generated text is factually grounded — it only checks if it "looks like" a real answer
- For PariShiksha, a GAN might produce fluent, confident-sounding answers that are completely wrong
- Grounded QA needs conditional generation (conditioned on specific textbook context), not unconditional sampling from a learned distribution
- The "adversarial" framing is about fooling a discriminator, but PariShiksha needs to NOT fool anyone — it needs verifiable accuracy

---

## 4. Encoder-Decoder vs Decoder-Only

**Question**: If you tried both T5 and Gemini, describe the differences you observed. Which produced more grounded answers? Which was more fluent? Which would you deploy for PariShiksha and why?

**Your Answer:**

_[Write here]_

---

## 5. What Your Evaluation Taught You

**Question**: Before running your evaluation, what did you expect your system's weaknesses to be? After running it, what surprised you? Be specific — name the question type or specific question that revealed something you didn't expect.

**Your Answer:**

_[Write here]_

---

## 6. The Messiness of Real Data

**Question**: What was the messiest part of the NCERT PDF extraction? How did you handle it? What cleaning steps did you add that you didn't initially plan for?

**Your Answer:**

_[Write here]_

---

## 7. Hindi Code-Switching

**Question**: How did your system handle Hindi-English mixed queries? What modifications, if any, would be needed to support this properly in production?

**Your Answer:**

_[Write here]_

---

## 8. Honest Self-Assessment

**Question**: On a scale of 1-5, rate your understanding of each topic after completing this project. Be honest — a "2" with an honest explanation is worth more than a dishonest "5".

| Topic | Rating (1-5) | Brief explanation |
|-------|-------------|-------------------|
| Tokenization | ___ | ___ |
| Attention mechanism | ___ | ___ |
| Encoder-decoder architecture | ___ | ___ |
| Transformer variants | ___ | ___ |
| GANs (conceptual) | ___ | ___ |
| Retrieval-augmented generation | ___ | ___ |
| Evaluation design | ___ | ___ |

---

## 9. What Would You Build Next Week?

**Question**: Next week you'll build a full vector-database RAG pipeline on top of this foundation. What specific improvements or changes would you make to your current system to prepare for that?

**Your Answer:**

_[Write here]_

---

## 10. Interview Preparation

**Question**: If an interviewer asked "Walk me through how your study assistant ensures it doesn't hallucinate," what would you say? Write your answer as if you were in an interview. Keep it under 3 minutes of speaking time (~400 words).

**Your Answer:**

_[Write here]_
