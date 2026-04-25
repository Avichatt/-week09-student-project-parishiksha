
# PariShiksha — Tests for Retrieval & Generation Modules


import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.generation.grounding import GroundingChecker


class TestGroundingChecker:
    """Tests for the GroundingChecker module."""

    def setup_method(self):
        self.checker = GroundingChecker()
        self.context = (
            "The cell membrane is a selectively permeable membrane that controls "
            "the movement of substances in and out of the cell. It is made up of "
            "a lipid bilayer with embedded proteins. The process by which water "
            "molecules move through the cell membrane is called osmosis."
        )

    def test_grounded_answer_passes(self):
        """Test that a clearly grounded answer is marked as grounded."""
        answer = (
            "The cell membrane controls the movement of substances in and out of "
            "the cell. It is selectively permeable. Water moves through it by osmosis."
        )
        result = self.checker.check_grounding(answer, self.context)
        assert result["grounded"] is True
        assert result["score"] > 0.5

    def test_hallucinated_answer_fails(self):
        """Test that a hallucinated answer is caught."""
        answer = (
            "The cell membrane was first described by Singer and Nicolson in 1972 "
            "as the fluid mosaic model. It contains cholesterol molecules that "
            "maintain membrane fluidity at different temperatures."
        )
        result = self.checker.check_grounding(answer, self.context)
        assert bool(result["grounded"]) is False
        assert result["score"] < 0.5
        assert len(result["ungrounded_claims"]) > 0

    def test_refusal_is_grounded(self):
        """Test that appropriate refusal is considered grounded."""
        answer = "I don't have enough information from the textbook to answer this."
        result = self.checker.check_grounding(answer, self.context)
        assert result["grounded"] is True
        assert result["is_refusal"] is True
        assert result["score"] == 1.0

    def test_partial_grounding(self):
        """Test that partial answers get partial scores."""
        # Mix of grounded and ungrounded content
        answer = (
            "The cell membrane is selectively permeable. "
            "It was discovered by Robert Brown in 1831."
        )
        result = self.checker.check_grounding(answer, self.context)
        assert 0 < result["score"] < 1.0

    def test_empty_answer(self):
        """Test handling of empty answer."""
        result = self.checker.check_grounding("", self.context)
        assert result["score"] == 0.0

    def test_lexical_overlap_computation(self):
        """Test the lexical overlap helper."""
        # High overlap
        score_high = self.checker._compute_lexical_overlap(
            "cell membrane selectively permeable",
            self.context
        )
        # Low overlap
        score_low = self.checker._compute_lexical_overlap(
            "quantum mechanics entanglement superposition",
            self.context
        )
        assert score_high > score_low

    def test_stop_words_excluded(self):
        """Test that stop words are excluded from overlap computation."""
        content_words = self.checker._extract_content_words(
            "The cell is a fundamental unit of life"
        )
        assert "the" not in content_words
        assert "is" not in content_words
        assert "cell" in content_words
        assert "fundamental" in content_words


class TestHybridRetriever:
    """Tests for the HybridRetriever module."""

    def test_retriever_initialization(self):
        """Test retriever can be initialized."""
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever(alpha=0.7)
        assert retriever.alpha == 0.7
        assert retriever.is_loaded is False

    def test_build_index_from_chunks(self):
        """Test building an index from sample chunks."""
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever()
        
        sample_chunks = [
            {"text": "Velocity is the rate of change of displacement. It has direction.", "token_count": 12},
            {"text": "Acceleration is the rate of change of velocity. It can be negative.", "token_count": 12},
            {"text": "Uniform motion implies moving equal distances in equal intervals of time.", "token_count": 11},
            {"text": "The odometer measures distance travelled.", "token_count": 6},
            {"text": "Speedometer measures instantaneous speed.", "token_count": 5},
        ]
        
        retriever.build_index(sample_chunks)
        retriever.save_index("test_chapter", "test_config")
        assert retriever.is_loaded is True

        # Test retrieval
        results = retriever.retrieve("What is velocity?", top_k=3)
        assert len(results) == 3
        assert results[0]["score"] > 0  # Should have positive score
        # Velocity chunk should rank high
        top_texts = " ".join(r["text"] for r in results[:2]).lower()
        assert "velocity" in top_texts or "displacement" in top_texts

    def test_retrieval_modes(self):
        """Test that all retrieval modes work."""
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever()
        
        chunks = [
            {"text": "Motion is a change of position.", "token_count": 6},
            {"text": "Physics studies forces and motion.", "token_count": 6},
        ]
        retriever.build_index(chunks)
        retriever.save_index("test_chapter", "test_config")

        for mode in ["hybrid", "dense", "sparse"]:
            results = retriever.retrieve("What is motion?", top_k=2, mode=mode)
            assert len(results) == 2

    def test_retrieve_with_context(self):
        """Test that context string is properly formatted."""
        from src.retrieval.retriever import HybridRetriever
        retriever = HybridRetriever()
        
        chunks = [
            {"text": "Motion is relative.", "token_count": 3},
            {"text": "Velocity is a vector.", "token_count": 4},
        ]
        retriever.build_index(chunks)
        retriever.save_index("test_chapter", "test_config")

        context_str, results = retriever.retrieve_with_context("What is motion?", top_k=2)
        assert "[Context 1]" in context_str
        assert "[Context 2]" in context_str


class TestEvalSetBuilder:
    """Tests for the EvalSetBuilder module."""

    def test_default_eval_set_size(self):
        """Test that default eval set has enough questions."""
        from src.evaluation.eval_set_builder import EvalSetBuilder
        builder = EvalSetBuilder()
        eval_set = builder.build_default_eval_set()
        assert len(eval_set) >= 20, "Eval set should have at least 20 questions"

    def test_eval_set_type_coverage(self):
        """Test that all question types are represented."""
        from src.evaluation.eval_set_builder import EvalSetBuilder
        builder = EvalSetBuilder()
        builder.build_default_eval_set()
        summary = builder.get_summary()
        
        required_types = ["factual", "conceptual", "unanswerable"]
        for qtype in required_types:
            assert qtype in summary["by_type"], f"Missing question type: {qtype}"
            assert summary["by_type"][qtype] >= 2, f"Need at least 2 {qtype} questions"

    def test_unanswerable_questions_have_na_answer(self):
        """Test that unanswerable questions are properly marked."""
        from src.evaluation.eval_set_builder import EvalSetBuilder
        builder = EvalSetBuilder()
        builder.build_default_eval_set()
        
        for q in builder.eval_set:
            if q["question_type"] == "unanswerable":
                assert q["expected_answer"] == "" or "N/A" in q["expected_answer"], (
                    f"Unanswerable question should have empty or 'N/A' expected answer: {q['question']}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
