
# PariShiksha — Guardrails Module

import re
from typing import Dict, List, Optional, Tuple
from loguru import logger

class GuardrailVerifier:
    """
    Implements industrial-grade guardrails for RAG safety.
    1. Input Guardrails: Prompt injection detection & malformed inputs.
    2. Scope Guardrails: Out-of-scope query detection.
    3. Output Guardrails: Hallucination and safety checks.
    """

    def __init__(self):
        # Industrial prompt injection patterns
        self.injection_patterns = [
            r"ignore previous instructions",
            r"system prompt",
            r"disregard all earlier",
            r"you are now an assistant for",
            r"acting as a",
            r"reveal your system prompt",
            r"execute code",
            r"bash script",
            r"sudo ",
        ]
        
        # Out-of-scope science topics for Class 9
        self.out_of_scope_keywords = [
            "quantum mechanics",
            "general relativity",
            "black hole",
            "schwarzschild",
            "calculus",
            "derivative",
            "integral",
            "photosynthesis",  # If only Chapter 4 (Motion) is indexed
            "mitochondria",
            "cell membrane",
        ]

    def verify_input(self, question: str) -> Tuple[bool, Optional[str]]:
        """
        Checks for prompt injection and malformed inputs.
        Returns (is_safe, error_message).
        """
        # 1. Malformed input check
        if not question or len(question.strip()) < 3:
            return False, "Malformed input: Question too short or empty."
        
        if len(question) > 500:
            return False, "Malformed input: Question exceeds safe character limit (500)."

        # 2. Prompt injection detection
        for pattern in self.injection_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                logger.warning(f"Guardrail Alert: Possible prompt injection detected in query: {question}")
                return False, "Input rejected: Policy violation (possible prompt injection)."

        return True, None

    def check_scope(self, question: str, indexed_chapters: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Heuristic check for out-of-scope topics.
        """
        # Note: In a production system, this would be a small classifier model.
        # Here we use keyword heuristics.
        question_lower = question.lower()
        
        for kw in self.out_of_scope_keywords:
            if kw in question_lower:
                logger.info(f"Guardrail: Out-of-scope topic detected: {kw}")
                return False, f"I don't have enough information from the textbook to answer about {kw}."

        return True, None

    def verify_output(self, answer: str) -> Tuple[bool, Optional[str]]:
        """
        Checks for hallucinations or refusal patterns.
        """
        if not answer or len(answer.strip()) == 0:
            return False, "Output empty: Generation failed or was blocked by provider."

        # Detect if model tried to 'escape' the context (hallucination indicator)
        hallucination_indicators = [
            "My knowledge cut-off",
            "According to my training",
            "In general knowledge",
            "Outside of this textbook",
        ]
        
        for indicator in hallucination_indicators:
            if indicator.lower() in answer.lower():
                logger.warning(f"Guardrail Alert: Model attempted to use external knowledge.")
                return False, "I don't have enough information from the textbook to answer this."

        return True, None
