
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
        Build the default evaluation set for NCERT Class 9 Science Chapter 4
        (Describing Motion Around Us).
        """
        self.eval_set = []

        # --- FACTUAL QUESTIONS ---
        self.add_question(
            question="What is the name of the ancient Indian treatise from the 5th century CE that mentions the concept of speed?",
            question_type="factual",
            expected_answer="The treatise is Aryabhatiya.",
            source_section="India's Scientific Contributions",
            expected_keywords=["Aryabhatiya"],
            chapter="chapter_4",
            difficulty="easy",
        )

        self.add_question(
            question="In the context of motion in a straight line, how are the two possible directions of motion represented?",
            question_type="factual",
            expected_answer="They are represented by plus (+) and minus (-) signs.",
            source_section="Describing position",
            expected_keywords=["plus", "minus", "+", "-"],
            chapter="chapter_4",
            difficulty="easy",
        )

        self.add_question(
            question="In Example 4.1 from Ganitakaumudi regarding the two postmen, what is the initial distance between them?",
            question_type="factual",
            expected_answer="The initial distance between them is 210 yojanas.",
            source_section="India's Scientific Contributions",
            expected_keywords=["210", "yojanas"],
            chapter="chapter_4",
            difficulty="medium",
        )

        # --- CONCEPTUAL QUESTIONS ---
        self.add_question(
            question="What is the condition for the distance travelled by an object to be equal to the magnitude of its displacement?",
            question_type="conceptual",
            expected_answer="The distance travelled and the magnitude of displacement are equal if the object moves in a straight line without turning back, i.e., it moves in one direction only.",
            source_section="Distance travelled and displacement",
            expected_keywords=["straight line", "without turning back", "one direction"],
            chapter="chapter_4",
            difficulty="medium",
        )

        self.add_question(
            question="What happens to the average velocity as the time interval becomes infinitesimally small?",
            question_type="conceptual",
            expected_answer="When the time interval becomes infinitesimally small, the average value of velocity approaches a fixed value called the instantaneous velocity.",
            source_section="Ready to Go Beyond",
            expected_keywords=["instantaneous velocity", "infinitesimally small", "approaches"],
            chapter="chapter_4",
            difficulty="medium",
        )

        # --- APPLICATION QUESTIONS ---
        self.add_question(
            question="If an athlete runs 100 meters forward on a straight track and then runs 100 meters back to the starting point, what are the total distance travelled and the displacement?",
            question_type="application",
            expected_answer="The total distance travelled is 200 meters, but the displacement is 0 meters since the starting and stopping positions are the same.",
            source_section="Distance travelled and displacement",
            expected_keywords=["200 meters", "0 meters"],
            chapter="chapter_4",
            difficulty="hard",
        )

        self.add_question(
            question="Sarang swims from one end of a 25 m pool to the other end and back to his starting point in 50 seconds. What is his average speed and average velocity?",
            question_type="application",
            expected_answer="His average speed is 1 m/s (50m/50s), and his average velocity is 0 m/s because his displacement is zero.",
            source_section="Average speed and average velocity",
            expected_keywords=["1 m s-1", "0 m s-1", "average speed", "average velocity"],
            chapter="chapter_4",
            difficulty="hard",
        )

        # --- UNANSWERABLE QUESTIONS ---
        self.add_question(
            question="How does Albert Einstein's theory of general relativity explain the curvature of spacetime caused by a black hole?",
            question_type="unanswerable",
            expected_answer="N/A — General Theory of Relativity and black holes are not covered in the opening motion chapter.",
            source_section="",
            expected_keywords=[],
            chapter="chapter_4",
            difficulty="hard",
        )

        # --- HINDI CODE-SWITCHED QUESTIONS ---
        self.add_question(
            question="Motion in a straight line ko aur kis naam se jaana jaata hai, according to the textbook?",
            question_type="hindi_codeswitched",
            expected_answer="It is also called linear motion.",
            source_section="Motion in a Straight Line",
            expected_keywords=["linear motion"],
            chapter="chapter_4",
            difficulty="medium",
        )

        logger.info(f"Built Motion (Ch 4) eval set with {len(self.eval_set)} questions")
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
