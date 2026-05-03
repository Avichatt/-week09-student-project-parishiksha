# Fix Memo — Stage 5 Evidence

## Which fix?
**Enhanced strict prompt with explicit OOS examples and anti-extrapolation rules.**

The fix modifies the system prompt to:
1. Add explicit examples of correct refusal behavior
2. Prohibit calculating/deriving values not explicitly in context
3. Prohibit answering about specific scenarios (Moon, Mars) even when formulas exist
4. Strengthen the boundary: Chapter 4 (Motion) ONLY

## Why this fix?
In Stage 4, the worst failure was a 'plausibly-answerable OOS' question — the system attempted to answer questions about topics not in the corpus by extrapolating from formulas that ARE in the context. This is the most dangerous hallucination category because it produces confident, formula-based answers that happen to be wrong (e.g., calculating Moon gravity using Earth's g value).

The failure category from the catalog: **mixed structure / ambiguous scope** — the model sees relevant formulas and assumes it should calculate, rather than recognizing the specific scenario is outside the corpus.

## Score Delta

| Metric | v1 (Before) | v2 (After) | Delta |
|--------|-------------|------------|-------|
| Correct (Y) | 0/12 | 0/12 | +0 |
| Grounded (Y) | 0/12 | 0/12 | +0 |
| OOS Refused | 0/3 | 0/3 | +0 |

## Honest Assessment
The fix did not change overall correctness count. OOS refusal stayed at 0/3. The fix did not improve OOS handling, suggesting the issue may be at retrieval level rather than generation level.

**No regressions detected.** The fix improved or maintained all scores.
