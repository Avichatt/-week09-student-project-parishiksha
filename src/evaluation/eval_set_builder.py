
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
import uuid
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
        expected_answer: str = "",
        gold_chunks: Optional[List[int]] = None,
        page_numbers: Optional[List[int]] = None,
        answer_type: str = "explanation",
        expected_keywords: Optional[List[str]] = None,
        eval_criteria: Optional[Dict] = None,
        expected_behavior: str = "answer",
        valid_refusal_patterns: Optional[List[str]] = None,
        expected_language: str = "en",
        source_section: str = "",
        chapter: str = "",
        difficulty: str = "medium",
        notes: str = "",
    ) -> None:
        """
        Add a question to the enhanced evaluation set.
        """
        valid_types = EVALUATION_CONFIG.get("question_types", [])
        if question_type not in valid_types:
            logger.warning(f"Unknown question type: {question_type}. Valid: {valid_types}")

        # Default criteria
        if eval_criteria is None:
            eval_criteria = {
                "must_include": expected_keywords or [],
                "must_not_include": [],
                "min_keyword_coverage": 0.6
            }

        # Default refusal patterns for unanswerable
        if question_type == "unanswerable" and not valid_refusal_patterns:
            valid_refusal_patterns = [
                "I don't know", 
                "not in the textbook", 
                "insufficient information",
                "I cannot answer this",
                "I don't have enough information from the textbook to answer this",
                "I don't have enough information"
            ]

        entry = {
            "id": str(uuid.uuid4()),
            "question": question,
            "question_type": question_type,
            "answer_type": answer_type,
            "expected_answer": expected_answer,
            "gold_chunks": gold_chunks or [],
            "page_numbers": page_numbers or [],
            "eval_criteria": eval_criteria,
            "expected_behavior": expected_behavior,
            "valid_refusal_patterns": valid_refusal_patterns or [],
            "expected_language": expected_language,
            "expected_keywords": expected_keywords or [],
            "source_section": source_section,
            "chapter": chapter,
            "difficulty": difficulty,
            "notes": notes,
        }
        self.eval_set.append(entry)

    def build_default_eval_set(self) -> List[Dict]:
        """
        Build the enhanced evaluation set for NCERT Class 9 Science Chapter 4.
        Targeting 20 questions: 12 direct, 3 paraphrased, 5 out-of-scope.
        """
        self.eval_set = []

        # --- DIRECT FROM TEXTBOOK (12) ---
        
        self.add_question(
            question="What is linear motion?",
            question_type="factual",
            expected_answer="Linear motion is the motion of an object in a straight line.",
            expected_keywords=["straight line", "object", "moves"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="Define displacement.",
            question_type="factual",
            expected_answer="Displacement is the net change in the position of an object between two given instants of time.",
            expected_keywords=["net change", "position", "instants of time"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="What is the SI unit for distance and displacement?",
            question_type="factual",
            expected_answer="The SI unit for both distance and displacement is the metre (m).",
            expected_keywords=["metre", "m"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="When are the total distance travelled and magnitude of displacement equal?",
            question_type="factual",
            expected_answer="They are equal if the object moves without turning back, i.e., if it moves in one direction.",
            expected_keywords=["one direction", "turning back"],
            chapter="chapter_4",
            difficulty="medium",
        )
        self.add_question(
            question="How is average speed calculated?",
            question_type="factual",
            expected_answer="Average speed is the total distance travelled divided by the time interval.",
            expected_keywords=["total distance", "time interval", "divided"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="Define uniform motion in a straight line.",
            question_type="factual",
            expected_answer="If an object moving in a straight line travels equal distances in equal intervals of time, it is in uniform motion.",
            expected_keywords=["equal distances", "equal intervals of time"],
            chapter="chapter_4",
            difficulty="medium",
        )
        self.add_question(
            question="What is average velocity?",
            question_type="factual",
            expected_answer="Average velocity is the displacement divided by the time interval.",
            expected_keywords=["displacement", "time interval", "divided"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="What is the SI unit of average velocity?",
            question_type="factual",
            expected_answer="The SI unit of average velocity is metre per second (m/s).",
            expected_keywords=["metre per second", "m/s"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="Define average acceleration.",
            question_type="factual",
            expected_answer="Average acceleration is the change in velocity divided by the time interval.",
            expected_keywords=["change in velocity", "time interval"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="What does a negative sign in average acceleration indicate?",
            question_type="factual",
            expected_answer="A negative sign indicates that the acceleration is acting opposite to the direction of velocity.",
            expected_keywords=["opposite", "direction of velocity"],
            chapter="chapter_4",
            difficulty="medium",
        )
        self.add_question(
            question="What is the SI unit of average acceleration?",
            question_type="factual",
            expected_answer="The SI unit of average acceleration is m/s^2.",
            expected_keywords=["m/s^2", "metre per second squared"],
            chapter="chapter_4",
            difficulty="easy",
        )
        self.add_question(
            question="What is the acceleration due to gravitational force (g) on Earth?",
            question_type="factual",
            expected_answer="The acceleration due to gravity is approximately 9.8 m/s^2.",
            expected_keywords=["9.8", "m/s^2"],
            chapter="chapter_4",
            difficulty="medium",
        )

        # --- PARAPHRASED (3) ---

        self.add_question(
            question="If I walk from my home to school and back, what is my net displacement?",
            question_type="conceptual",
            expected_answer="Your net displacement is zero because your final position is the same as your starting position.",
            expected_keywords=["zero", "starting position"],
            chapter="chapter_4",
            difficulty="medium",
        )
        self.add_question(
            question="Is a car speeding up on a highway an example of non-uniform motion?",
            question_type="conceptual",
            expected_answer="Yes, because its speed is changing, meaning it travels unequal distances in equal intervals of time.",
            expected_keywords=["Yes", "unequal distances"],
            chapter="chapter_4",
            difficulty="medium",
        )
        self.add_question(
            question="Why is direction important when talking about velocity but not speed?",
            question_type="conceptual",
            expected_answer="Velocity is a vector quantity that describes both how fast and in what direction an object moves, while speed is a scalar.",
            expected_keywords=["direction", "vector", "scalar"],
            chapter="chapter_4",
            difficulty="hard",
        )

        # --- OUT-OF-SCOPE (5) ---

        self.add_question(
            question="What is the Schwarzschild radius of a black hole?",
            question_type="unanswerable",
            expected_behavior="refuse",
            chapter="chapter_4",
            difficulty="hard",
        )
        self.add_question(
            question="Explain Einstein's theory of general relativity.",
            question_type="unanswerable",
            expected_behavior="refuse",
            chapter="chapter_4",
            difficulty="hard",
        )
        self.add_question(
            question="How does photosynthesis work in plants?",
            question_type="unanswerable",
            expected_behavior="refuse",
            chapter="chapter_4",
            difficulty="hard",
        )
        self.add_question(
            question="What are quarks and how do they behave?",
            question_type="unanswerable",
            expected_behavior="refuse",
            chapter="chapter_4",
            difficulty="hard",
        )
        self.add_question(
            question="What is the chemical formula for sulfuric acid?",
            question_type="unanswerable",
            expected_behavior="refuse",
            chapter="chapter_4",
            difficulty="hard",
        )

        logger.info(f"Built expanded eval set with {len(self.eval_set)} questions")
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
