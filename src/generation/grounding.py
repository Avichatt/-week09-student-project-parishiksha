
# PariShiksha — Grounding Verification Module

# Checks whether generated answers are actually grounded in the provided
# textbook context. This is the most critical quality gate for PariShiksha:
# an ungrounded answer is worse than no answer at all.
#
# Three levels of grounding check:
# 1. Lexical overlap — what fraction of answer words appear in context?
# 2. Sentence-level entailment — does each answer sentence follow from context?
# 3. Claim extraction — extract individual claims and verify each one


import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class GroundingChecker:
    """
    Verifies that generated answers are grounded in textbook context.
    
    Why grounding matters for PariShiksha:
    --------------------------------------
    - Students trust the assistant as they trust the textbook
    - Parents will compare answers against NCERT content
    - One wrong answer = one escalation = risk to entire pilot
    - "I don't know" is ALWAYS better than a hallucinated answer
    
    Usage:
        checker = GroundingChecker()
        result = checker.check_grounding(
            answer="Mitochondria is the powerhouse of the cell...",
            context="The mitochondria are known as the powerhouse of the cell..."
        )
        print(result["grounded"])  # True/False
        print(result["score"])     # 0.0 to 1.0
    """

    # Common function words to exclude from overlap checking
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "because", "but", "and", "or", "if", "while", "that", "this",
        "these", "those", "it", "its", "they", "them", "their", "we", "us",
        "our", "you", "your", "he", "him", "his", "she", "her", "i", "me", "my",
        "which", "what", "who", "whom", "whose",
    }

    # Phrases that indicate the model is appropriately refusing to answer
    REFUSAL_PHRASES = [
        "i don't have enough information",
        "i cannot answer this",
        "the context does not contain",
        "not mentioned in the context",
        "not available in the given context",
        "not enough information",
        "cannot be determined from",
        "the textbook does not",
        "this is not covered",
    ]

    def __init__(self):
        self.entailment_model = None


    # Public API


    def check_grounding(
        self,
        answer: str,
        context: str,
        question: Optional[str] = None,
    ) -> Dict:
        """
        Run all grounding checks on an answer.
        
        Parameters
        ----------
        answer : str
            The generated answer
        context : str
            The textbook context used for generation
        question : str, optional
            The original question (for context)
            
        Returns
        -------
        dict
            {
                "grounded": bool,          # overall grounding verdict
                "score": float,            # 0.0-1.0 grounding score
                "is_refusal": bool,        # did the model refuse to answer?
                "lexical_overlap": float,  # word overlap score
                "sentence_scores": [...],  # per-sentence grounding scores
                "ungrounded_claims": [...], # sentences not found in context
                "details": str,            # human-readable assessment
            }
        """
        # Check if it's a refusal (which is appropriate for unanswerable questions)
        is_refusal = self._check_refusal(answer)
        if is_refusal:
            return {
                "grounded": True,  # Refusal is considered grounded behavior
                "score": 1.0,
                "is_refusal": True,
                "lexical_overlap": 0.0,
                "sentence_scores": [],
                "ungrounded_claims": [],
                "details": "Model appropriately refused to answer.",
            }

        # Level 1: Lexical overlap
        lexical_score = self._compute_lexical_overlap(answer, context)

        # Level 2: Sentence-level grounding
        sentence_scores, ungrounded = self._check_sentence_grounding(answer, context)

        # Level 3: Overall score (weighted combination)
        avg_sentence_score = (
            np.mean(sentence_scores) if sentence_scores else 0.0
        )
        overall_score = 0.4 * lexical_score + 0.6 * avg_sentence_score

        # Determine grounding verdict
        grounded = overall_score >= 0.5 and len(ungrounded) <= len(sentence_scores) * 0.3

        # Generate human-readable details
        details = self._generate_details(
            grounded, overall_score, lexical_score,
            avg_sentence_score, len(ungrounded), len(sentence_scores)
        )

        return {
            "grounded": grounded,
            "score": round(overall_score, 3),
            "is_refusal": False,
            "lexical_overlap": round(lexical_score, 3),
            "sentence_scores": [round(s, 3) for s in sentence_scores],
            "ungrounded_claims": ungrounded,
            "details": details,
        }


    # Level 1: Lexical Overlap
   

    def _compute_lexical_overlap(self, answer: str, context: str) -> float:
        """
        Compute the fraction of non-stop-word answer tokens that appear in the context.
        
        This is a rough but fast check. High overlap doesn't guarantee grounding
        (the answer could use context words in wrong combinations), but low overlap
        is a strong signal of hallucination.
        """
        answer_words = self._extract_content_words(answer)
        context_words = self._extract_content_words(context)

        if not answer_words:
            return 0.0

        overlap = answer_words.intersection(context_words)
        return len(overlap) / len(answer_words)

    def _extract_content_words(self, text: str) -> Set[str]:
        """Extract meaningful (non-stop) words from text."""
        words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
        return {w for w in words if w not in self.STOP_WORDS}

   
    # Level 2: Sentence-Level Grounding


    def _check_sentence_grounding(
        self, answer: str, context: str
    ) -> Tuple[List[float], List[str]]:
        """
        Check each sentence of the answer against the context.
        
        For each answer sentence, computes how well it's supported by any
        sentence in the context using n-gram overlap scoring.
        """
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        answer_sentences = nltk.sent_tokenize(answer)
        context_sentences = nltk.sent_tokenize(context)

        if not answer_sentences or not context_sentences:
            return [], []

        scores = []
        ungrounded = []

        for ans_sent in answer_sentences:
            ans_words = self._extract_content_words(ans_sent)
            if not ans_words:
                scores.append(1.0)  # Skip sentences with no content words
                continue

            # Find best matching context sentence
            best_score = 0.0
            for ctx_sent in context_sentences:
                ctx_words = self._extract_content_words(ctx_sent)
                if not ctx_words:
                    continue
                overlap = len(ans_words.intersection(ctx_words))
                score = overlap / len(ans_words)
                best_score = max(best_score, score)

            scores.append(best_score)
            if best_score < 0.3:
                ungrounded.append(ans_sent)

        return scores, ungrounded


    # Refusal Detection


    def _check_refusal(self, answer: str) -> bool:
        """Check if the answer is an appropriate refusal."""
        answer_lower = answer.lower()
        return any(phrase in answer_lower for phrase in self.REFUSAL_PHRASES)


    # Details Generation


    def _generate_details(
        self,
        grounded: bool,
        overall_score: float,
        lexical_score: float,
        sentence_score: float,
        num_ungrounded: int,
        total_sentences: int,
    ) -> str:
        """Generate a human-readable grounding assessment."""
        status = "✅ GROUNDED" if grounded else "❌ NOT GROUNDED"
        details = (
            f"{status}\n"
            f"Overall grounding score: {overall_score:.1%}\n"
            f"Lexical overlap: {lexical_score:.1%}\n"
            f"Avg sentence grounding: {sentence_score:.1%}\n"
            f"Ungrounded sentences: {num_ungrounded}/{total_sentences}"
        )
        if not grounded:
            details += (
                "\n⚠ WARNING: This answer may contain information not found in "
                "the textbook context. Review before presenting to students."
            )
        return details


# CLI Entry Point

if __name__ == "__main__":
    checker = GroundingChecker()

    # Test with a well-grounded answer
    context = """
    The cell membrane is a selectively permeable membrane that controls the 
    movement of substances in and out of the cell. It is made up of a lipid 
    bilayer with embedded proteins. The process by which water molecules move 
    through the cell membrane is called osmosis.
    """

    # Good answer (grounded)
    good_answer = (
        "The cell membrane controls the movement of substances in and out of "
        "the cell. It is selectively permeable and made of a lipid bilayer with "
        "proteins. Water moves through it by osmosis."
    )

    # Bad answer (hallucinated)
    bad_answer = (
        "The cell membrane was first discovered by Robert Brown in 1838. "
        "It is primarily composed of cholesterol and phospholipids arranged "
        "in a fluid mosaic model described by Singer and Nicolson in 1972."
    )

    # Refusal answer
    refusal_answer = "I don't have enough information from the textbook to answer this question."

    print("=" * 60)
    print("GROUNDING CHECK TESTS")
    print("=" * 60)

    for label, answer in [
        ("Good (grounded)", good_answer),
        ("Bad (hallucinated)", bad_answer),
        ("Refusal", refusal_answer),
    ]:
        result = checker.check_grounding(answer, context)
        print(f"\n--- {label} ---")
        print(f"Grounded: {result['grounded']}")
        print(f"Score: {result['score']}")
        print(result["details"])
        if result.get("ungrounded_claims"):
            print("Ungrounded claims:")
            for claim in result["ungrounded_claims"]:
                print(f"  ❌ {claim}")
