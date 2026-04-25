
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
            "Uniform motion is a type of motion where an object covers equal "
            "distances in equal intervals of time. In contrast, non-uniform "
            "motion means unequal distances in equal intervals of time. The "
            "rate of motion can be calculated by finding the average speed."
        )

    def test_grounded_answer_passes(self):
        """Test that a clearly grounded answer is marked as grounded."""
        answer = (
            "Uniform motion occurs when an object covers equal distances in equal "
            "time intervals. Finding the average speed helps calculate the rate of motion."
        )
        result = self.checker.check_grounding(answer, self.context)
        assert result["grounded"] is True
        assert result["score"] > 0.5

    def test_hallucinated_answer_fails(self):
        """Test that a hallucinated answer is caught."""
        answer = (
            "The theory of relativity was first described by Albert Einstein in 1905. "
            "It explains that the speed of light is constant in a vacuum and "
            "time dilation occurs at high speeds."
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
            "Uniform motion means covering equal distances in equal time intervals. "
            "It was first studied extensively by Isaac Newton in the 17th century."
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
            "uniform motion equal distances intervals",
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
            "The motion is a fundamental concept of physics"
        )
        assert "the" not in content_words
        assert "is" not in content_words
        assert "motion" in content_words
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
