
# PariShiksha — Evaluation Set Builder

# Builds the question-answer evaluation set for the study assistant.
# This is Stage 4 of the project, and arguably the most important:
# "If you can't measure it, you can't improve it."

# The eval set includes 5 question types:
# 1. Factual — Direct fact recall ("Who discovered cells?")
# 2. Conceptual — Understanding ("Why do cells need energy?")
# 3. Application — Apply concept ("If a cell loses its nucleus...")
# 4. Unanswerable — Not in textbook (tests hallucination resistance)
# 5. Hindi code-switched — Hindi-English mixed queries


import json
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import EVALUATION_DATA_DIR, EVALUATION_CONFIG


class EvalSetBuilder:
    """
    Builds and manages the evaluation question set.
    
    Why eval design matters:
    -----------------------
    - An eval set that only tests easy factual questions will make your
      system look better than it is. Parents won't ask only easy questions.
    - Unanswerable questions are the most important category: they test
      whether the system hallucinates or honestly says "I don't know."
    - Hindi code-switched queries are realistic for PariShiksha's students
      who naturally switch between Hindi and English.
    
    Usage:
        builder = EvalSetBuilder()
        
        # Load the pre-built eval set
        eval_set = builder.load_eval_set()
        
        # Or build from scratch
        builder.add_question(
            question="What is the function of mitochondria?",
            question_type="factual",
            expected_answer="Mitochondria generate energy...",
            source_section="Cell Organelles",
            expected_keywords=["energy", "ATP", "powerhouse"],
        )
        builder.save_eval_set()
    """

    def __init__(self, output_dir: Path = EVALUATION_DATA_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.eval_set: List[Dict] = []

    # Building the Eval Set


    def add_question(
        self,
        question: str,
        question_type: str,
        expected_answer: str,
        source_section: str = "",
        expected_keywords: Optional[List[str]] = None,
        chapter: str = "",
        difficulty: str = "medium",
        notes: str = "",
    ) -> None:
        """
        Add a question to the evaluation set.
        
        Parameters
        ----------
        question : str
            The question text
        question_type : str
            One of: factual, conceptual, application, unanswerable, hindi_codeswitched
        expected_answer : str
            The reference answer (from textbook for answerable, "N/A" for unanswerable)
        source_section : str
            Which textbook section contains the answer
        expected_keywords : list of str
            Key terms that should appear in a correct answer
        chapter : str
            Chapter identifier (e.g., "chapter_5")
        difficulty : str
            easy, medium, hard
        notes : str
            Additional notes for evaluation
        """
        valid_types = EVALUATION_CONFIG.get("question_types", [])
        if question_type not in valid_types:
            logger.warning(f"Unknown question type: {question_type}. Valid: {valid_types}")

        entry = {
            "id": len(self.eval_set) + 1,
            "question": question,
            "question_type": question_type,
            "expected_answer": expected_answer,
            "source_section": source_section,
            "expected_keywords": expected_keywords or [],
            "chapter": chapter,
            "difficulty": difficulty,
            "notes": notes,
        }
        self.eval_set.append(entry)

    def build_default_eval_set(self) -> List[Dict]:
        """
        Build the default evaluation set for NCERT Class 9 Science Chapter 1
        (Exploration: Entering the World of Secondary Science).
        """
        self.eval_set = []

        # --- FACTUAL QUESTIONS ---
        self.add_question(
            question="What do the magnifying glass and compass symbolize in scientific exploration?",
            question_type="factual",
            expected_answer="The magnifying glass symbolises careful observation—noticing patterns. The compass reminds us that exploration needs direction—choosing models and asking the right questions.",
            source_section="Symbolism",
            expected_keywords=["magnifying glass", "observation", "compass", "direction"],
            chapter="chapter_1",
            difficulty="easy",
        )

        self.add_question(
            question="Who is the scientist mentioned as simplifying stars as a hot gas?",
            question_type="factual",
            expected_answer="The physicist Meghnad Saha treated the matter in stars as a hot gas to explain star colours.",
            source_section="Meet a Scientist",
            expected_keywords=["Meghnad Saha", "stars", "hot gas"],
            chapter="chapter_1",
            difficulty="easy",
        )

        self.add_question(
            question="What happened in the airplane fuel miscalculation incident?",
            question_type="factual",
            expected_answer="A passenger aircraft ran out of fuel because the ground crew used units of pounds (lb) per litre instead of kilograms (kg) per litre.",
            source_section="Ready to Go Beyond",
            expected_keywords=["fuel", "miscalculation", "pounds", "kilograms"],
            chapter="chapter_1",
            difficulty="medium",
        )

        # --- CONCEPTUAL QUESTIONS ---
        self.add_question(
            question="Why does science use simplified models of real systems?",
            question_type="conceptual",
            expected_answer="Science uses models to make sense of complexity by focusing only on what is most important for a given question and deliberately ignoring certain details.",
            source_section="Models in Science",
            expected_keywords=["models", "complexity", "focus", "ignoring details"],
            chapter="chapter_1",
            difficulty="medium",
        )

        self.add_question(
            question="What is the difference between a scientific law and a theory?",
            question_type="conceptual",
            expected_answer="A law describes a regular pattern observed in nature. A theory goes further and provides an explanation of why those patterns occur based on evidence.",
            source_section="Threads of Curiosity",
            expected_keywords=["law", "pattern", "theory", "explanation"],
            chapter="chapter_1",
            difficulty="medium",
        )

        # --- APPLICATION QUESTIONS ---
        self.add_question(
            question="Estimate how many breaths a person takes in a day according to the textbook.",
            question_type="application",
            expected_answer="An average person takes roughly 20 thousand breaths a day at rest (12-15 breaths per minute).",
            source_section="Estimation",
            expected_keywords=["20 thousand", "breaths", "day"],
            chapter="chapter_1",
            difficulty="hard",
        )

        self.add_question(
            question="If a viral claim says food is harmful during an eclipse, how can scientific thinking disprove it?",
            question_type="application",
            expected_answer="By asking scientific questions: Does temperature change significantly? Does food go bad in a shadow? One concludes no mechanism supports such a claim as an eclipse is just a play of shadows.",
            source_section="Checking viral claims",
            expected_keywords=["scientific questions", "shadows", "no mechanism"],
            chapter="chapter_1",
            difficulty="hard",
        )

        # --- UNANSWERABLE QUESTIONS ---
        self.add_question(
            question="What is the chemical equation for photosynthesis?",
            question_type="unanswerable",
            expected_answer="N/A — Photosynthesis is not covered in the opening 'Exploration' chapter.",
            source_section="",
            expected_keywords=[],
            chapter="chapter_1",
            difficulty="hard",
        )

        # --- HINDI CODE-SWITCHED QUESTIONS ---
        self.add_question(
            question="Science mein approximate estimation ki kya importance hai?",
            question_type="hindi_codeswitched",
            expected_answer="Estimation help karta hai intuition built karne mein, errors detect karne mein, and check karne mein ki answer sense bana raha hai ya nahi.",
            source_section="Estimation",
            expected_keywords=["estimation", "intuition", "sense"],
            chapter="chapter_1",
            difficulty="medium",
        )

        logger.info(f"Built Exploration (Ch 1) eval set with {len(self.eval_set)} questions")
        return self.eval_set


    # I/O
 

    def save_eval_set(self, filename: str = "eval_set.json") -> Path:
        """Save evaluation set to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_set, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved eval set ({len(self.eval_set)} questions) to {output_path}")
        return output_path

    def load_eval_set(self, filename: str = "eval_set.json") -> List[Dict]:
        """Load evaluation set from JSON."""
        input_path = self.output_dir / filename
        if not input_path.exists():
            logger.warning(f"Eval set not found at {input_path}. Building default...")
            return self.build_default_eval_set()
        
        with open(input_path, "r", encoding="utf-8") as f:
            self.eval_set = json.load(f)
        logger.info(f"Loaded eval set: {len(self.eval_set)} questions")
        return self.eval_set

    def get_summary(self) -> Dict:
        """Get distribution summary of the evaluation set."""
        if not self.eval_set:
            return {"total": 0}

        type_counts = {}
        difficulty_counts = {}
        chapter_counts = {}

        for q in self.eval_set:
            qtype = q.get("question_type", "unknown")
            type_counts[qtype] = type_counts.get(qtype, 0) + 1

            diff = q.get("difficulty", "unknown")
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

            ch = q.get("chapter", "unknown")
            chapter_counts[ch] = chapter_counts.get(ch, 0) + 1

        return {
            "total": len(self.eval_set),
            "by_type": type_counts,
            "by_difficulty": difficulty_counts,
            "by_chapter": chapter_counts,
        }



# CLI Entry Point

if __name__ == "__main__":
    builder = EvalSetBuilder()
    builder.build_default_eval_set()
    builder.save_eval_set()

    summary = builder.get_summary()
    print("\n" + "=" * 60)
    print("EVALUATION SET SUMMARY")
    print("=" * 60)
    print(f"Total questions: {summary['total']}")
    print(f"\nBy type:")
    for qtype, count in summary["by_type"].items():
        print(f"  {qtype}: {count}")
    print(f"\nBy difficulty:")
    for diff, count in summary["by_difficulty"].items():
        print(f"  {diff}: {count}")
    print(f"\nBy chapter:")
    for ch, count in summary["by_chapter"].items():
        print(f"  {ch}: {count}")

    print(f"\nSample questions:")
    for q in builder.eval_set[:3]:
        print(f"\n  [{q['question_type']}] {q['question']}")
        print(f"  Expected: {q['expected_answer'][:80]}...")
