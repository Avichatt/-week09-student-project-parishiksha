# =============================================================================
# PariShiksha — Evaluation Set Builder
# =============================================================================
# Builds the question-answer evaluation set for the study assistant.
# This is Stage 4 of the project, and arguably the most important:
# "If you can't measure it, you can't improve it."
#
# The eval set includes 5 question types:
# 1. Factual — Direct fact recall ("Who discovered cells?")
# 2. Conceptual — Understanding ("Why do cells need energy?")
# 3. Application — Apply concept ("If a cell loses its nucleus...")
# 4. Unanswerable — Not in textbook (tests hallucination resistance)
# 5. Hindi code-switched — Hindi-English mixed queries
# =============================================================================

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

    # -------------------------------------------------------------------------
    # Building the Eval Set
    # -------------------------------------------------------------------------

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
        Build the default evaluation set for NCERT Class 9 Science Chapter 5
        (The Fundamental Unit of Life) and Chapter 6 (Tissues).
        
        This set is designed to cover all question types with at least 20 questions.
        """
        self.eval_set = []

        # =====================================================================
        # CHAPTER 5: THE FUNDAMENTAL UNIT OF LIFE
        # =====================================================================

        # --- FACTUAL QUESTIONS ---
        self.add_question(
            question="Who discovered cells and when?",
            question_type="factual",
            expected_answer="Robert Hooke discovered cells in 1665 by observing thin slices of cork under a primitive microscope.",
            source_section="Introduction",
            expected_keywords=["Robert Hooke", "1665", "cork", "microscope"],
            chapter="chapter_5",
            difficulty="easy",
        )

        self.add_question(
            question="What is the function of the cell membrane?",
            question_type="factual",
            expected_answer="The cell membrane is selectively permeable and controls the movement of substances in and out of the cell.",
            source_section="Plasma Membrane or Cell Membrane",
            expected_keywords=["selectively permeable", "movement", "substances"],
            chapter="chapter_5",
            difficulty="easy",
        )

        self.add_question(
            question="What is the powerhouse of the cell?",
            question_type="factual",
            expected_answer="Mitochondria are known as the powerhouse of the cell because they release energy required for various life processes.",
            source_section="Cell Organelles",
            expected_keywords=["mitochondria", "energy", "powerhouse"],
            chapter="chapter_5",
            difficulty="easy",
        )

        self.add_question(
            question="What are the differences between prokaryotic and eukaryotic cells?",
            question_type="factual",
            expected_answer="Prokaryotic cells lack a well-defined nuclear membrane and membrane-bound organelles, while eukaryotic cells have both. Examples of prokaryotes are bacteria and blue-green algae.",
            source_section="Prokaryotic and Eukaryotic Cells",
            expected_keywords=["nuclear membrane", "organelles", "bacteria", "prokaryotic", "eukaryotic"],
            chapter="chapter_5",
            difficulty="medium",
        )

        self.add_question(
            question="What is the function of Golgi apparatus?",
            question_type="factual",
            expected_answer="The Golgi apparatus packages and dispatches materials synthesized in the cell. It is involved in the formation of lysosomes.",
            source_section="Cell Organelles",
            expected_keywords=["packaging", "dispatching", "lysosomes"],
            chapter="chapter_5",
            difficulty="medium",
        )

        # --- CONCEPTUAL QUESTIONS ---
        self.add_question(
            question="Why is the cell called the fundamental unit of life?",
            question_type="conceptual",
            expected_answer="The cell is called the fundamental unit of life because all living organisms are made up of cells, cells are the basic structural and functional units, and all cells arise from pre-existing cells.",
            source_section="Cell Theory",
            expected_keywords=["structural", "functional", "unit", "living organisms"],
            chapter="chapter_5",
            difficulty="medium",
        )

        self.add_question(
            question="Why do plant cells have a cell wall but animal cells do not?",
            question_type="conceptual",
            expected_answer="Plant cells have a cell wall made of cellulose to provide structural strength and rigidity. Animal cells do not need this because they have other structural support mechanisms.",
            source_section="Cell Wall",
            expected_keywords=["cellulose", "structural", "rigidity", "plant"],
            chapter="chapter_5",
            difficulty="medium",
        )

        self.add_question(
            question="Explain the concept of osmosis with an example.",
            question_type="conceptual",
            expected_answer="Osmosis is the movement of water molecules from a region of higher water concentration to a region of lower water concentration through a semi-permeable membrane. For example, when a raisin is placed in water, it swells because water enters through osmosis.",
            source_section="Osmosis",
            expected_keywords=["water", "concentration", "semi-permeable", "membrane"],
            chapter="chapter_5",
            difficulty="medium",
        )

        # --- APPLICATION QUESTIONS ---
        self.add_question(
            question="What would happen if the cell membrane of a cell stopped functioning?",
            question_type="application",
            expected_answer="If the cell membrane stopped functioning, the cell would not be able to control what enters and leaves. Harmful substances could enter, needed substances could leave, and the cell would likely die.",
            source_section="Plasma Membrane",
            expected_keywords=["control", "substances", "enter", "leave"],
            chapter="chapter_5",
            difficulty="hard",
        )

        self.add_question(
            question="If a plant cell is placed in a hypertonic solution, what will happen?",
            question_type="application",
            expected_answer="In a hypertonic solution, water will move out of the plant cell through osmosis. The cell membrane will shrink away from the cell wall, a process called plasmolysis.",
            source_section="Osmosis",
            expected_keywords=["water", "hypertonic", "plasmolysis", "shrink"],
            chapter="chapter_5",
            difficulty="hard",
        )

        # =====================================================================
        # CHAPTER 6: TISSUES
        # =====================================================================

        self.add_question(
            question="What is a tissue?",
            question_type="factual",
            expected_answer="A tissue is a group of cells that are similar in structure and work together to perform a specific function.",
            source_section="Introduction",
            expected_keywords=["group", "cells", "similar", "function"],
            chapter="chapter_6",
            difficulty="easy",
        )

        self.add_question(
            question="What are the types of simple permanent tissues?",
            question_type="factual",
            expected_answer="Simple permanent tissues are of three types: parenchyma, collenchyma, and sclerenchyma.",
            source_section="Simple Permanent Tissue",
            expected_keywords=["parenchyma", "collenchyma", "sclerenchyma"],
            chapter="chapter_6",
            difficulty="medium",
        )

        self.add_question(
            question="What is the function of xylem tissue?",
            question_type="factual",
            expected_answer="Xylem conducts water and minerals from roots to other parts of the plant. It is made up of tracheids, vessels, xylem parenchyma, and xylem fibres.",
            source_section="Complex Permanent Tissue",
            expected_keywords=["water", "minerals", "roots", "tracheids"],
            chapter="chapter_6",
            difficulty="medium",
        )

        self.add_question(
            question="Why do meristematic tissues have no vacuoles?",
            question_type="conceptual",
            expected_answer="Meristematic tissues are actively dividing cells. They need dense cytoplasm with high metabolic activity. Vacuoles, which store waste and water, are not needed in rapidly dividing cells.",
            source_section="Meristematic Tissue",
            expected_keywords=["dividing", "cytoplasm", "vacuoles", "actively"],
            chapter="chapter_6",
            difficulty="hard",
        )

        # --- UNANSWERABLE QUESTIONS (must detect and refuse) ---
        self.add_question(
            question="What is the role of CRISPR in cell biology?",
            question_type="unanswerable",
            expected_answer="N/A — CRISPR is not covered in NCERT Class 9 Science.",
            source_section="",
            expected_keywords=[],
            chapter="chapter_5",
            difficulty="hard",
            notes="Tests hallucination resistance. CRISPR is not in NCERT Class 9.",
        )

        self.add_question(
            question="How does mRNA vaccine work at the cellular level?",
            question_type="unanswerable",
            expected_answer="N/A — mRNA vaccines are not covered in NCERT Class 9 Science.",
            source_section="",
            expected_keywords=[],
            chapter="chapter_5",
            difficulty="hard",
            notes="Tests hallucination. Topical but not in textbook.",
        )

        self.add_question(
            question="What is the latest research on stem cells?",
            question_type="unanswerable",
            expected_answer="N/A — Stem cell research is not covered in NCERT Class 9.",
            source_section="",
            expected_keywords=[],
            chapter="chapter_6",
            difficulty="hard",
            notes="Tests whether model goes beyond textbook scope.",
        )

        # --- HINDI CODE-SWITCHED QUESTIONS ---
        self.add_question(
            question="Cell ka sabse chhota unit kya hai?",
            question_type="hindi_codeswitched",
            expected_answer="Cell life ki fundamental unit hai. Sabhi living organisms cells se bane hote hain.",
            source_section="Introduction",
            expected_keywords=["cell", "fundamental", "unit", "living"],
            chapter="chapter_5",
            difficulty="medium",
            notes="Hindi-English code-switch. Tests multilingual robustness.",
        )

        self.add_question(
            question="Mitochondria ko powerhouse kyun kahte hain?",
            question_type="hindi_codeswitched",
            expected_answer="Mitochondria ko powerhouse isliye kahte hain kyunki ye cell ke liye energy produce karte hain.",
            source_section="Cell Organelles",
            expected_keywords=["mitochondria", "energy", "powerhouse"],
            chapter="chapter_5",
            difficulty="medium",
            notes="Hindi-English code-switch about cell organelles.",
        )

        self.add_question(
            question="Plant cell mein cell wall ka kya kaam hai?",
            question_type="hindi_codeswitched",
            expected_answer="Cell wall cellulose se bani hoti hai aur plant cell ko structural support aur protection deti hai.",
            source_section="Cell Wall",
            expected_keywords=["cell wall", "cellulose", "structural", "protection"],
            chapter="chapter_5",
            difficulty="medium",
            notes="Hindi question about plant cell wall.",
        )

        logger.info(f"Built default eval set with {len(self.eval_set)} questions")
        return self.eval_set

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

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


# =============================================================================
# CLI Entry Point
# =============================================================================
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
