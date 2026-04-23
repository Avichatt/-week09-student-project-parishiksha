# PariShiksha — Evaluation Report Template

> Fill this in after running the evaluation pipeline (`python main.py --stage evaluate`)

---

## 1. Evaluation Set Summary

| Metric | Value |
|--------|-------|
| Total questions | ___ |
| Factual questions | ___ |
| Conceptual questions | ___ |
| Application questions | ___ |
| Unanswerable questions | ___ |
| Hindi code-switched questions | ___ |

---

## 2. Stage 1 Results: Tokenizer Comparison

### 2.1 Scientific Term Fragmentation

| Term | BERT tokens | T5 tokens | GPT-2 tokens | BioBERT tokens |
|------|------------|-----------|--------------|----------------|
| photosynthesis | ___ | ___ | ___ | ___ |
| mitochondria | ___ | ___ | ___ | ___ |
| endoplasmic reticulum | ___ | ___ | ___ | ___ |
| ... | | | | |

### 2.2 Tokenizer Recommendation

- **Best for chunking**: ___ (rationale: ___)
- **Most compact**: ___ (chars/token: ___)
- **Worst fragmentation**: ___ (avg tokens/term: ___)

---

## 3. Stage 2 Results: Chunking Experiment

### 3.1 Configuration Comparison

| Strategy | Chunk Size | Num Chunks | Avg Tokens | Min | Max | Std |
|----------|-----------|------------|------------|-----|-----|-----|
| fixed_token | 128 | ___ | ___ | ___ | ___ | ___ |
| fixed_token | 256 | ___ | ___ | ___ | ___ | ___ |
| fixed_token | 512 | ___ | ___ | ___ | ___ | ___ |
| sentence_based | 128 | ___ | ___ | ___ | ___ | ___ |
| sentence_based | 256 | ___ | ___ | ___ | ___ | ___ |
| sentence_based | 512 | ___ | ___ | ___ | ___ | ___ |
| semantic_paragraph | 128 | ___ | ___ | ___ | ___ | ___ |
| semantic_paragraph | 256 | ___ | ___ | ___ | ___ | ___ |
| semantic_paragraph | 512 | ___ | ___ | ___ | ___ | ___ |

### 3.2 Best Configuration

- **Selected**: ___ at ___ tokens
- **Rationale**: ___

---

## 4. Stage 3 Results: Retrieval & Generation

### 4.1 Retrieval Quality

| Retrieval Mode | Avg Top-1 Score | Keyword Recall |
|---------------|-----------------|----------------|
| Dense only | ___ | ___ |
| Sparse only | ___ | ___ |
| Hybrid (α=0.7) | ___ | ___ |

### 4.2 Generation Quality (Gemini)

| Question Type | Avg Quality Score | Grounded % | Notes |
|--------------|-------------------|------------|-------|
| Factual | ___ | ___ | |
| Conceptual | ___ | ___ | |
| Application | ___ | ___ | |
| Unanswerable | ___ (refusal rate) | ___ | |
| Hindi code-switched | ___ | ___ | |

### 4.3 Generation Quality (Flan-T5)

| Question Type | Avg Quality Score | Grounded % | Notes |
|--------------|-------------------|------------|-------|
| Factual | ___ | ___ | |
| Conceptual | ___ | ___ | |
| Application | ___ | ___ | |
| Unanswerable | ___ (refusal rate) | ___ | |
| Hindi code-switched | ___ | ___ | |

---

## 5. Grounding Analysis

### 5.1 Hallucination Detection

| Metric | Gemini | T5 |
|--------|--------|-----|
| Answers grounded | ___/total | ___/total |
| Avg grounding score | ___ | ___ |
| Unanswerable correctly refused | ___/3 | ___/3 |
| Ungrounded claims detected | ___ | ___ |

### 5.2 Failure Cases

Document the most interesting failure cases here:

1. **Question**: ___
   - **Expected**: ___
   - **Generated**: ___
   - **Why it failed**: ___

2. **Question**: ___
   - **Expected**: ___
   - **Generated**: ___
   - **Why it failed**: ___

---

## 6. Overall Assessment

### What works well:
- 

### What needs improvement:
- 

### Surprises:
- 

### If I had one more week:
- 
