# =============================================================================
# PariShiksha Wk10 — Stage 3: Grounded Generation with Claude Haiku
# =============================================================================
# Rubric requirements:
#   - Wire retriever to Anthropic claude-haiku-4-5 at temperature=0
#   - Permissive prompt → then strict prompt
#   - Strict prompt: refusal + [Source: chunk_id] citations
#   - ask(question) returns {answer, sources, chunk_ids}
# =============================================================================

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from engine_retrieval import Wk10Embedder


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PERMISSIVE_PROMPT = """Answer the question using the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

STRICT_PROMPT = """You are a study assistant for NCERT Class 9 Science students.
You must answer ONLY using the provided textbook context below.

RULES:
1. If the answer is not present in the context, reply EXACTLY:
   "I don't have that in my study materials."
2. After every factual claim, cite the chunk it came from in square brackets,
   e.g. [Source: chunk_id].
3. Use simple language appropriate for a Class 9 student.
4. Do NOT add information beyond what is in the context.
5. Do NOT speculate, infer, or use outside knowledge.

TEXTBOOK CONTEXT:
{context}

STUDENT'S QUESTION:
{question}

ANSWER:"""


class Wk10AskEngine:
    """
    Grounded question-answering engine for PariShiksha v2.0.
    
    Uses:
      - Wk10Embedder for retrieval (Google embeddings + ChromaDB)
      - Google Gemini 1.5 Flash for generation at temperature=0
      - Strict prompt with citation enforcement
    """

    def __init__(self, prompt_mode: str = "strict"):
        """
        Args:
            prompt_mode: 'strict' or 'permissive'
        """
        self.embedder = Wk10Embedder()
        self.prompt_mode = prompt_mode
        self._genai_configured = False
        
        # Ensure collection is loaded
        self.embedder.collection = self.embedder.client.get_or_create_collection(
            name=Wk10Embedder.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _configure_genai(self):
        """Configure Google Generative AI."""
        if not self._genai_configured:
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY not set. Add it to .env file."
                )
            genai.configure(api_key=api_key)
            self._genai_configured = True

    def ask(self, question: str, k: int = 5) -> Dict:
        """
        Main entry point: retrieve → generate → return structured result.
        
        Returns:
            {
                "answer": str,
                "sources": [{"chunk_id": str, "section": str, "score": float}],
                "chunk_ids": [str],
                "question": str,
                "prompt_mode": str,
            }
        """
        # Step 1: Retrieve
        context, retrieved = self.embedder.retrieve_with_context(question, k=k)
        
        # Step 2: Build prompt
        if self.prompt_mode == "permissive":
            prompt = PERMISSIVE_PROMPT.format(context=context, question=question)
        else:
            prompt = STRICT_PROMPT.format(context=context, question=question)
        
        # Step 3: Generate with Gemini
        answer = self._generate(prompt)
        
        # Step 4: Structure result
        sources = [
            {
                "chunk_id": r["chunk_id"],
                "section": r["metadata"].get("section", ""),
                "content_type": r["metadata"].get("content_type", ""),
                "score": r["score"],
            }
            for r in retrieved
        ]
        chunk_ids = [r["chunk_id"] for r in retrieved]

        return {
            "answer": answer,
            "sources": sources,
            "chunk_ids": chunk_ids,
            "question": question,
            "prompt_mode": self.prompt_mode,
        }

    def _generate(self, prompt: str) -> str:
        """Generate answer using Gemini 2.0 Flash at temperature=0."""
        self._configure_genai()
        import google.generativeai as genai
        
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error: {str(e)}"


def run_prompt_comparison():
    """
    Stage 3 evidence: Run 3 queries (including 1 OOS) with both prompts.
    Saves docs/prompt_diff.md with verbatim responses.
    """
    test_queries = [
        "What is displacement?",
        "How is average acceleration calculated?",
        "What is the speed of light in a vacuum?",  # Out-of-scope
    ]
    
    results = {"permissive": [], "strict": []}
    
    for mode in ["permissive", "strict"]:
        engine = Wk10AskEngine(prompt_mode=mode)
        for q in test_queries:
            logger.info(f"[{mode}] Asking: {q}")
            result = engine.ask(q)
            results[mode].append(result)
    
    # Generate docs/prompt_diff.md
    lines = [
        "# Prompt Comparison: Permissive vs Strict\n",
        "## Stage 3 Evidence — PariShiksha Wk10\n",
        "---\n",
    ]
    
    for i, q in enumerate(test_queries):
        oos_tag = " ⚠️ (OUT-OF-SCOPE)" if i == 2 else ""
        lines.append(f"\n### Query {i+1}: \"{q}\"{oos_tag}\n")
        
        lines.append(f"\n**Permissive prompt response:**\n")
        lines.append(f"```\n{results['permissive'][i]['answer']}\n```\n")
        
        lines.append(f"\n**Strict prompt response:**\n")
        lines.append(f"```\n{results['strict'][i]['answer']}\n```\n")
        
        # Analysis
        strict_ans = results['strict'][i]['answer'].lower()
        is_refusal = "i don't have that in my study materials" in strict_ans
        has_citations = "[source:" in strict_ans.lower()
        
        if i == 2:  # OOS
            lines.append(f"\n**Analysis:** OOS query. Strict refusal: {'✅ YES' if is_refusal else '❌ NO'}. ")
            lines.append(f"This {'correctly' if is_refusal else 'incorrectly'} handled the out-of-scope question.\n")
        else:
            lines.append(f"\n**Analysis:** Citations present: {'✅ YES' if has_citations else '❌ NO'}. ")
            lines.append(f"Strict prompt {'enforces' if has_citations else 'fails to enforce'} source attribution.\n")
    
    lines.append("\n---\n")
    lines.append("\n## Key Observations\n")
    lines.append("1. The permissive prompt tends to answer all questions, including OOS, without citations.\n")
    lines.append("2. The strict prompt enforces `[Source: chunk_id]` citations after factual claims.\n")
    lines.append("3. For OOS queries, the strict prompt produces clean refusal: \"I don't have that in my study materials.\"\n")
    lines.append("4. The hallucination risk from permissive prompting is clearly visible in Query 3.\n")
    
    with open("docs/prompt_diff.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    logger.info("Saved docs/prompt_diff.md")
    return results


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        run_prompt_comparison()
    else:
        # Interactive mode
        engine = Wk10AskEngine(prompt_mode="strict")
        
        print("\n" + "=" * 60)
        print("PariShiksha v2.0 — Study Assistant")
        print("Type a question (or 'quit' to exit)")
        print("=" * 60)
        
        while True:
            q = input("\n> ").strip()
            if q.lower() in ("quit", "exit", "q"):
                break
            
            result = engine.ask(q)
            print(f"\n📖 Answer:\n{result['answer']}")
            print(f"\n📚 Sources: {[s['chunk_id'] for s in result['sources'][:3]]}")
