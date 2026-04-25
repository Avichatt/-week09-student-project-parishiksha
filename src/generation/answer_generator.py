
# PariShiksha — Answer Generation Module

# Generates grounded answers using retrieved context chunks.
# Supports two architectural families:
#   1. Decoder-only (Gemini) — for high-quality generation
#   2. Encoder-decoder (Flan-T5) — for comparison in Stretch/Advanced
#
# The key principle: the model must ONLY use the provided context.
# Any answer that goes beyond the textbook is a hallucination,
# and in PariShiksha's context, a hallucination could mean a parent
# pulls their child from the program.


import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import GENERATION_CONFIG, GEMINI_API_KEY
from src.generation.guardrails import GuardrailVerifier


class AnswerGenerator:
    """
    Generates answers grounded in NCERT textbook context.
    
    Architecture comparison (Stretch/Advanced):
    -------------------------------------------
    - Decoder-only (Gemini): Strong at natural, fluent generation.
      Uses full context window. Better at following complex instructions.
      Risk: more prone to adding plausible-sounding but unsupported claims.
    
    - Encoder-decoder (Flan-T5): Encoder reads context, decoder generates.
      More "extractive" in nature — tends to copy from context.
      Risk: can be choppy, but less likely to hallucinate.
    
    Usage:
        generator = AnswerGenerator()
        
        # Using Gemini
        answer = generator.generate_answer(
            question="What is the function of mitochondria?",
            context="[Retrieved context here]",
            model_type="gemini"
        )
        
        # Using T5
        answer = generator.generate_answer(
            question="What is the function of mitochondria?",
            context="[Retrieved context here]",
            model_type="t5"
        )
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or GENERATION_CONFIG
        self.gemini_model = None
        self.t5_model = None
        self.t5_tokenizer = None
        self.guardrails = GuardrailVerifier()
        self._setup_logging()

    def _setup_logging(self):
        """Configure structured logging for generation tracking."""
        self.generation_log = []


    # Gemini (Decoder-Only)
 

    def _init_gemini(self) -> None:
        """Initialize Gemini API client."""
        if self.gemini_model is not None:
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model_name = self.config.get("gemini_model", "gemini-1.5-flash")
            self.gemini_model = genai.GenerativeModel(model_name)
            logger.info(f"Initialized Gemini: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            raise

    def _generate_gemini(self, question: str, context: str) -> Dict:
        """
        Generate answer using Gemini (decoder-only).
        
        Prompt structure:
        1. System instruction (grounding rules)
        2. Context (retrieved chunks)
        3. Question
        4. Output format instructions
        """
        self._init_gemini()

        system_prompt = self.config.get("system_prompt", "")
        temperature = self.config.get("gemini_temperature", 0.3)
        max_tokens = self.config.get("gemini_max_tokens", 512)

        prompt = f"""{system_prompt}

TEXTBOOK CONTEXT:
{context}

STUDENT'S QUESTION:
{question}

INSTRUCTIONS:
1. Answer using ONLY the textbook context above.
2. If the answer is not in the context, say so clearly.
3. Use simple language suitable for Class 9-10 students.
4. If the question involves a concept, explain it step by step.
5. Quote relevant parts of the textbook when helpful.

ANSWER:"""

        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                },
            )
            answer_text = response.text.strip()
            
            result = {
                "answer": answer_text,
                "model": "gemini",
                "model_name": self.config.get("gemini_model"),
                "prompt_tokens": len(prompt.split()),  # approximate
                "temperature": temperature,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            result = {
                "answer": f"Error: {str(e)}",
                "model": "gemini",
                "status": "error",
                "error": str(e),
            }

        return result


    # Flan-T5 (Encoder-Decoder)
 

    def _init_t5(self) -> None:
        """Initialize Flan-T5 model and tokenizer."""
        if self.t5_model is not None:
            return

        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            
            model_name = self.config.get("t5_model", "google/flan-t5-base")
            logger.info(f"Loading T5 model: {model_name}")
            
            self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            logger.info(f"Loaded T5: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load T5: {e}")
            raise

    def _generate_t5(self, question: str, context: str) -> Dict:
        """
        Generate answer using Flan-T5 (encoder-decoder).
        
        T5 is a text-to-text model: we format the input as a task prompt
        and the model generates the output. The encoder processes the context,
        and the decoder generates the answer conditioned on it.
        
        Key difference from decoder-only:
        - Encoder has bidirectional attention (sees full context at once)
        - Decoder generates left-to-right
        - Tends to be more extractive (copies from context more)
        """
        self._init_t5()

        max_length = self.config.get("t5_max_length", 256)
        num_beams = self.config.get("t5_num_beams", 4)

        # T5-style prompt format
        prompt = (
            f"Answer the following question based only on the given context. "
            f"If the answer is not in the context, say 'I cannot answer this from the given context.'\n\n"
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        try:
            # Tokenize (truncate if too long for T5's context window)
            inputs = self.t5_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,      # T5-base has 512 token limit
                truncation=True,
            )

            # Generate
            outputs = self.t5_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

            answer_text = self.t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            result = {
                "answer": answer_text.strip(),
                "model": "t5",
                "model_name": self.config.get("t5_model"),
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": outputs.shape[1],
                "num_beams": num_beams,
                "status": "success",
            }
        except Exception as e:
            logger.error(f"T5 generation failed: {e}")
            result = {
                "answer": f"Error: {str(e)}",
                "model": "t5",
                "status": "error",
                "error": str(e),
            }

        return result


    # Unified Interface


    def generate_answer(
        self,
        question: str,
        context: str,
        source_chunks: Optional[List[Dict]] = None,
        model_type: str = "gemini",
        teacher_mode: bool = True,
    ) -> Dict:
        """
        Industrial-grade generation with guardrails and teacher-mode citations.
        """
        # 1. Input Guardrails
        is_safe, error_msg = self.guardrails.verify_input(question)
        if not is_safe:
            return {
                "answer": error_msg,
                "model": model_type,
                "status": "blocked_input"
            }

        # 2. Scope Guardrails
        in_scope, scope_msg = self.guardrails.check_scope(question, [])
        if not in_scope:
             return {
                "answer": scope_msg,
                "model": model_type,
                "status": "out_of_scope"
            }

        # 3. Context Window Safety
        max_ctx_tokens = self.config.get("max_context_tokens", 1500)
        max_ctx_chars = max_ctx_tokens * 4
        
        if len(context) > max_ctx_chars:
            logger.warning(f"Context too long ({len(context)} chars). Truncating for safety.")
            context = context[:max_ctx_chars] + "\n[Context Truncated for Safety...]"

        # 4. Generation
        if model_type == "gemini":
            result = self._generate_gemini(question, context)
        elif model_type == "t5":
            result = self._generate_t5(question, context)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gemini' or 't5'.")

        # 5. Output Guardrails
        is_output_safe, output_error = self.guardrails.verify_output(result["answer"])
        if not is_output_safe:
            result["answer"] = output_error
            result["status"] = "blocked_output"

        # 6. Teacher Mode (Citations)
        if teacher_mode and source_chunks and result["status"] == "success":
            citations = []
            for i, chunk in enumerate(source_chunks[:3]): # Top 3 citations
                page = chunk.get("metadata", {}).get("page_number", "N/A")
                section = chunk.get("metadata", {}).get("section", "General")
                citations.append(f"Source {i+1}: Page {page}, Section: {section}")
            
            if citations:
                result["answer"] += "\n\n📚 **Textbook References:**\n- " + "\n- ".join(citations)

        # Log generation
        result["question"] = question
        result["context_length_chars"] = len(context)
        self.generation_log.append(result)

        return result

    def compare_models(
        self,
        question: str,
        context: str,
    ) -> Dict:
        """
        Generate answers from both Gemini and T5 for comparison.
        
        This is the core of the Advanced tier: seeing how decoder-only
        and encoder-decoder architectures produce different answers
        from the same retrieved context.
        """
        gemini_result = self.generate_answer(question, context, model_type="gemini")
        t5_result = self.generate_answer(question, context, model_type="t5")

        return {
            "question": question,
            "context_length": len(context),
            "gemini": gemini_result,
            "t5": t5_result,
        }

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def save_generation_log(self, output_path: Path) -> None:
        """Save all generation results for evaluation."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.generation_log, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.generation_log)} generation results to {output_path}")



# CLI Entry Point

if __name__ == "__main__":
    generator = AnswerGenerator()

    # Example usage
    test_context = """
    Motion is defined as a change in the position of an object over time.
    Velocity is the rate of change of displacement, whereas speed is the rate of change of distance.
    Acceleration is the rate of change of velocity.
    Isaac Newton formulated three laws of motion:
    1. An object remains at rest or in uniform motion unless acted upon by a net external force.
    2. The rate of change of momentum of a body is directly proportional to the applied force.
    3. For every action, there is an equal and opposite reaction.
    """

    test_question = "What are Newton's three laws of motion?"

    print("=" * 60)
    print("ANSWER GENERATION TEST")
    print("=" * 60)
    print(f"\nQuestion: {test_question}")
    print(f"Context length: {len(test_context)} chars")

    try:
        result = generator.generate_answer(test_question, test_context, model_type="gemini")
        print(f"\n[Gemini] Status: {result['status']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"\nGemini failed: {e}")

    try:
        result = generator.generate_answer(test_question, test_context, model_type="t5")
        print(f"\n[T5] Status: {result['status']}")
        print(f"Answer: {result['answer']}")
    except Exception as e:
        print(f"\nT5 failed: {e}")
