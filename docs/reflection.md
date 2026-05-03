# PariShiksha v2.0 — Reflection & Industrial Analysis

## Part A — Your implementation specifics

### A1. Chunking decisions, with evidence
My final chunking strategy was designed to be **content-type aware**, prioritizing the structural integrity of pedagogical units (like worked examples and activities) over arbitrary token counts.

**Parameters**:
- **Target Size**: 250 tokens (cl100k_base via `tiktoken`).
- **Overlap**: 0 tokens. I intentionally chose zero overlap because the content-aware splitter identifies natural boundaries (headings, "Example X", "Activity Y"). Adding overlap would create redundant citations for the same pedagogical unit.
- **Splitter Rules**: 
  - `prose`: Standard recursive character splitting within section boundaries.
  - `worked_example`: Atomic preservation. If a block starts with "Example", it is kept as a single unit up to 500 tokens to ensure the problem, solution, and answer stay together.
  - `question_or_exercise`: Similar to worked examples, preserving the full context of an "Activity" or "Exercise" set.

**Sample Chunks from `wk10_chunks.json`**:
1.  **chunk_id: `674452c9c5f1` (content_type: `prose`)**:
    - *Content*: The introduction to Chapter 4, describing butterflies, snakes, and the general wonder of motion. 
    - *Why*: It consists purely of narrative text without specific problem-solving structures.
2.  **chunk_id: `a01abdb940cb` (content_type: `worked_example`)**:
    - *Content*: "Example 4.1: Consider two postmen... Answer: both postmen will meet each other after 15 days."
    - *Why*: It follows the "Example X.X" pattern and includes the "Answer:" keyword. My chunker identified this as a critical unit to preserve so the retriever wouldn't return the question without the answer.
3.  **chunk_id: `33b3dfb3a478` (content_type: `question_or_exercise`)**:
    - *Content*: "Activity 4.1: Let us analyse... Table 4.1: Distance travelled and displacement of the ball."
    - *Why*: It was caught by the "Activity" regex. It includes a table structure that was flattened into text but kept together so the student can see the full experimental setup.

### A2. The chunk that surprised you
**chunk_id: `43a37cd2ba33`**
- *Text*: `[4.1.2 Distance travelled and displacement] The direction of displacement is specified from the position at the first instant towards the position at the second instant. To describe the total distan...`
- *Surprise*: I expected this to be a clean `prose` chunk about displacement. However, because the NCERT layout uses sidebars for "Note" and "Ready to Go Beyond," the parser injected these sidebars into the middle of the text stream. 
- *Heuristic Failure*: My heuristic for "prose" assumes a continuous flow of narrative. It missed the fact that the text "Physical quantities which can be specified by just their numerical value are called scalars" was actually a sidebar note, not part of the main paragraph. This resulted in a "noisy" chunk that contains two different levels of information (main text + definitions).

### A3. Loader choice
I stayed with **PyMuPDF (fitz)** because of its high speed and robust handling of font encodings in NCERT PDFs. 
- **Behavior Difference**: Compared to `pdfplumber`, PyMuPDF is much faster for batch processing (crucial for v2.0 industrialization). However, it is "layout-blind" in its default text extraction mode, which leads to the sidebar-merging issue described in A2. 
- **Testing**: I tested this by comparing the extracted text from `iesc104.pdf` using both libraries. While `pdfplumber` attempts to isolate tables better, it frequently fails on the multi-column sections of the physics textbook, creating fragmented sentences. PyMuPDF's `get_text("text")` provides a more coherent flow for RAG, even if it brings along sidebar "noise."

---

## Part B — Numbers from your evaluation

### B1. Eval scores, raw
In my final v2.0 evaluation run:
- (a) **Correct**: 0/20 (Technical failure)
- (b) **Grounded**: 0/20 (Technical failure)
- (c) **Appropriate Refusals**: 0/5 (Technical failure)

**The number that bothered me most**: The **0/20** across the board.
This was due to a **429 Quota Exceeded** error from the Gemini API during the final automated run. This bothered me deeply because it exposed a massive single-point-of-failure in a "production-grade" pipeline. While the architecture (retrieval, RRF, reranking) is significantly more advanced than v1.0, it is currently useless without cloud connectivity. This realization shifted my focus from "model quality" to "system resilience"—in a real study assistant, a quota hit would mean hundreds of students cannot study.

### B2. The single worst question
- **Question**: "Calculate the value of g on the surface of the Moon."
- **Answer**: `Error: 429 You exceeded your current quota...`
- **Top-3 Retrieved Chunks**: `59e9c4e21e2e` (Earth g), `4536f2c9c1ac` (acceleration definition), `dd2f628bc6ba` (calculation steps).
- **Failure Mode**: **Ambiguous Scope / Hallucination Risk**. 
Even before the API error, this question was a failure because the retriever found "g on Earth" (9.8 m/s²) and the model, in its attempt to be helpful, would likely have used that value to "calculate" Moon gravity or simply answered 9.8 m/s² by mistake. The failure is that the system doesn't realize "Moon" is out-of-scope for a Chapter 4 (Motion) knowledge base.

### B3. RAGAS (Stretch)
Due to the Gemini quota constraints and environment issues with `scikit-network`, I was unable to produce a live RAGAS report for the final commit.
- **Estimated Metric**: If I were to judge based on manual traces, **Context Precision** would be high (thanks to Hybrid + Rerank), but **Faithfulness** would be the risk point. 
- **Inference**: A high precision/low faithfulness gap tells me that **retrieval is working**, but the **generation stage is too "smart"**—it's using internal weights to fill in gaps instead of sticking strictly to the retrieved context.

---

## Part C — The 30-second debugging story

### C1. The retrieved chunk that fooled me
- **Query**: "What is the definition of displacement?"
- **Chunk text (674452c9c5f1)**: `[Introduction] Describing Motion Around Us Chapter Everything in nature is in motion, from massive astronomical objects to subatomic particles. We have a wide variety of motion in nature...`
- **Score**: 0.7944 (Cosine Similarity)
- **Why it ranked top-1**: This is the chapter introduction. It mentions "displacement" in the very last sentence: "...you will learn about some more physical quantities, such as displacement, average velocity and average acceleration." The vector search saw the keyword "displacement" in a high-level summary chunk and ranked it top-1, even though the *actual* definition is in chunk `17bf7b9f714e` (ranked #2). This is a classic case where the "summary" of a topic has higher similarity than the "details" of the topic.

### C2. The bug that took you longest
The **UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'**. 
- **Time to fix**: 2 hours.
- **Initial attempt**: I tried changing the terminal's code page using `chcp 65001`. This didn't work because the Python `print()` statement in a background process still used the default `cp1252` encoding of the Windows environment.
- **Actual Fix**: I had to manually hunt down the emoji in the evaluation script and replace it with plain text ("SUCCESS:"). 
- **Fastest Path for Teammates**: Never use emojis in CLI tool outputs intended for Windows users; always use plain ASCII for logging.

### C3. The thing that still bothers me
The **Sidebar Noise**. 
It still bothers me that "Ready to Go Beyond" or "Curiosity" sidebars are interleaved with physics formulas. Why? Because it breaks the semantic coherence of the chunk. In Wk11, I would implement a **Layout-Aware PDF Parser** (likely using `layout-parser` or `Unstructured`) to extract these sidebars into their own metadata fields instead of polluting the main text.

---

## Part D — Architecture and tradeoffs

### D1. Why hybrid retrieval?
"Why do we need both? Pick one and ship."
**I pick Hybrid.**
- **Query where BM25 wins**: "Example 4.1". BM25 identifies the exact string match "Example 4.1" instantly. Dense retrieval often confuses it with other "Example" chunks because the semantic "vibe" of a worked example is similar across the whole book.
- **Query where Dense wins**: "If I walk to school and back home, what is my total displacement?" The student doesn't use the textbook definition. BM25 might fail if "walk" and "school" aren't in the text. Dense retrieval understands the *concept* of a round trip and finds the chunk defining displacement as "net change in position."

### D2. The CRAG / Self-RAG question
I would build **CRAG** in production when the cost of a wrong answer is higher than the latency of a web search (e.g., a medical assistant). It is **overkill** for a Class 9 Study Assistant where the textbook is the "absolute truth." 
**Would it have helped my worst failure?** (Moon gravity). **Yes.** If the system recognized the retrieval was low-quality/off-topic, CRAG could have searched Wikipedia for Moon's g-value. However, this violates the pedagogical constraint: we want the student to learn from the *curriculum*, not the general internet.

### D3. Honest pilot readiness
**Can we launch Monday? NO.**
1. **Verify Quota Resilience**: We need a fallback to a local LLM (like Llama 3) to prevent the "0/20" disaster I faced.
2. **Fix Sidebar Merging**: Current chunks are too messy for high-stakes exams.
3. **Refusal Calibration**: The "Moon Gravity" hallucination risk is still too high. I need to verify that OOS questions are refused 100% of the time based on real eval data from rows OOS1–OOS5.

---

## Part E — Effort and self-assessment

### E1. Effort rating: 9/10.
I am genuinely proud of the **Content-Type-Aware Chunker**. Moving from simple character splits to logical pedagogical splits (Prose vs Example vs Activity) feels like a real step toward an "industrial" mindset.

### E2. The gap between you and a stronger student
A stronger student probably spent more time on **Data Cleaning**. They likely used regex to strip out the page headers ("Describing Motion Around Us") and sidebar noise that I left in. I didn't do this because I prioritized the **Hybrid Retrieval** architecture over data-cleaning polish.

### E3. The Industry Pointer: Reranking.
In 6 months, I'd explore **Cross-Encoder Reranking** deeply. It is the single most effective way to solve the "Top-1 is a summary, not a definition" problem (C1). The first step would be fine-tuning a small MiniLM model on physics-specific query/chunk pairs.

### E4. Two more days
1. **First thing**: Integrate a **Local LLM (Ollama/Llama3)** to ensure the pipeline runs even when APIs fail.
2. **Last thing**: Build a **Streamlit UI** to let a real student test the citations. Order matters because a UI is useless if the engine is down.
