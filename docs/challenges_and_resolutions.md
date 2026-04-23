# 🛡️ PariShiksha — Challenges & Resolutions

This document outlines the major engineering and integration challenges encountered during the development of the PariShiksha RAG pipeline and how they were systematically resolved.

| Challenge | Impact | Solution |
| :--- | :--- | :--- |
| **Windows Dependency Conflicts** | `ragas` and other C++ based libraries failed to build on Windows due to missing compiler headers. | Removed `ragas` from `requirements.txt` and built a custom, lightweight version of the grounding and keyword-recall metrics using `NumPy` and `scikit-learn`. |
| **PDF Extraction Noise** | Raw extraction included headers ("SCIENCE", page numbers), footer copyrights, and "mojibake" (mangled characters like `â€™`). | Implemented a **7-stage regex-based cleaning pipeline** in `TextCleaner` that specifically targets NCERT formatting patterns and replaces mojibake with UTF-8 equivalents. |
| **Scientific Term Fragmentation** | Default tokenizers (like GPT-2) split complex terms like "Deoxyribonucleic" into too many fragments, diluting semantic meaning. | Developed a `TokenizerAnalyzer` to compare 4 tokenizers; chose **BERT-base** for chunking as it showed the highest high-fidelity "compression" for NCERT science vocabulary. |
| **Retrieval vs. Semantic Accuracy** | Dense search (SBERT) was good at meaning but often missed exact keywords like "Lysosomes" or "1665". | Engineered a **Hybrid Retriever** using an $\alpha$-weighted formula ($\alpha=0.7$) to combine Dense (SBERT) and Sparse (TF-IDF) similarity scores. |
| **Hallucination Control** | Gemini models occasionally added "common knowledge" (e.g., cell organelles not in the text) instead of sticking solely to the provided textbook context. | Developed a **multi-level Grounding Checker** that performs lexical overlap and sentence-level verification. If the answer exceeds a "foreign content" threshold, it is flagged as ungrounded. |
| **Unicode Console Crashes** | On Windows PowerShell, printing professional box-drawing characters (─, ┌) caused `UnicodeEncodeError` and crashed the pipeline. | Modified the `PariShikshaEvaluator` to use only standard ASCII characters (`=`, `-`, `|`) for console summaries, ensuring cross-platform stability. |
| **Deprecated Model Names (2026 Environment)** | The requested `gemini-1.5-flash` model was deprecated or missing in the target environment. | Systematically mapped available models using `genai.list_models()` and updated `config.py` to use `models/gemini-flash-latest` (stable 2.0+ series). |
| **API Quota Management** | High-frequency evaluation calls (20 questions with 6k+ chars context) hit free-tier rate limits, causing `429` errors. | Optimized the `AnswerGenerator` to handle retry logic and switched to the more efficient `flash-latest` model to minimize token pressure. |
| **Git History & Grading Rigor** | A single-commit "dump" would have failed the engineering discipline criteria for the PG Diploma grading. | Reconstructed the git history by clearing the `.git` metadata and performing **7 logical, stage-wise commits** to demonstrate a professional development lifecycle. |

---

### 🧠 Conclusion
The resolution of these challenges transformed the project from a simple "script" into a robust, cross-platform **engineering pipeline** capable of handling messy real-world data and strict pedagogical constraints.
