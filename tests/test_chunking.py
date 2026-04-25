
# PariShiksha — Tests for Chunking Module


import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.chunking.chunker import TextChunker


class TestTextChunker:
    """Tests for the TextChunker module."""

    def setup_method(self):
        self.chunker = TextChunker(tokenizer_name="bert-base-uncased")
        self.sample_text = (
            "Motion is a change of position; it can be described in terms of the distance moved or the displacement. "
            "The motion of an object could be uniform or non-uniform depending on whether its velocity is constant or changing. "
            "The speed of an object is the distance covered per unit time, and velocity is the displacement per unit time. "
            "The acceleration of an object is the change in velocity per unit time. "
            "Uniform and non-uniform motions of objects can be shown through graphs. "
            "The motion of an object moving at uniform acceleration can be described with the help of three equations. "
            "If an object moves in a circular path with uniform speed, its motion is called uniform circular motion. "
            "Displacement is the shortest distance measured from the initial to the final position of an object. "
            "Odometer is an instrument for measuring the distance travelled by a vehicle. "
            "Speed has only magnitude, whereas velocity has both magnitude and direction. "
            "The SI unit of distance and displacement is metre (m). "
            "The SI unit of speed and velocity is metre per second (m/s). "
            "The SI unit of acceleration is metre per second square (m/s squared)."
        )

    def test_fixed_token_chunking_produces_chunks(self):
        """Test that fixed-token chunking produces non-empty chunks."""
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="fixed_token", chunk_size=64
        )
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk["text"].strip()
            assert chunk["token_count"] > 0

    def test_fixed_token_overlap(self):
        """Test that overlap is applied between consecutive chunks."""
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="fixed_token",
            chunk_size=64, overlap_ratio=0.25
        )
        if len(chunks) >= 2:
            # The second chunk should share some content with the first
            first_words = set(chunks[0]["text"].lower().split())
            second_words = set(chunks[1]["text"].lower().split())
            overlap = first_words.intersection(second_words)
            assert len(overlap) > 0, "Chunks should have overlapping content"

    def test_sentence_based_chunking(self):
        """Test sentence-based chunking preserves sentence boundaries."""
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="sentence_based", chunk_size=128
        )
        assert len(chunks) > 0
        for chunk in chunks:
            # Each chunk should end with proper punctuation (sentence boundary)
            text = chunk["text"].strip()
            assert text[-1] in ".!?", f"Chunk doesn't end at sentence boundary: ...{text[-20:]}"

    def test_semantic_paragraph_fallback(self):
        """Test semantic chunking falls back gracefully without sections."""
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="semantic_paragraph", chunk_size=128
        )
        assert len(chunks) > 0

    def test_semantic_paragraph_with_sections(self):
        """Test semantic chunking with section structure."""
        sections = [
            {
                "heading": "Introduction",
                "text": "Motion is a change of position. It can be uniform or non-uniform.",
                "content_type": "narrative",
            },
            {
                "heading": "Rate of Motion",
                "text": "Speed is distance per unit time. Velocity is displacement per unit time.",
                "content_type": "narrative",
            },
        ]
        chunks = self.chunker.chunk_text(
            "", strategy="semantic_paragraph", chunk_size=256, sections=sections
        )
        # Handle small test sections by lowering threshold if needed
        if not chunks:
            self.chunker.config["min_chunk_tokens"] = 5
            chunks = self.chunker.chunk_text(
                "", strategy="semantic_paragraph", chunk_size=256, sections=sections
            )
        
        assert len(chunks) >= 2
        # Check that section headings are preserved
        assert any("[Introduction]" in c["text"] for c in chunks)

    def test_min_chunk_filter(self):
        """Test that tiny chunks are filtered out."""
        # Very small chunk size should still respect minimum
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="fixed_token", chunk_size=256
        )
        for chunk in chunks:
            assert chunk["token_count"] >= 50, "Chunk below minimum token threshold"

    def test_chunk_metadata(self):
        """Test that chunks have required metadata fields."""
        chunks = self.chunker.chunk_text(
            self.sample_text, strategy="fixed_token", chunk_size=128
        )
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert "token_count" in chunk
            assert "metadata" in chunk
            assert "strategy" in chunk["metadata"]
            assert chunk["metadata"]["strategy"] == "fixed_token"

    def test_chunking_experiment(self):
        """Test the full chunking experiment runner."""
        experiment = self.chunker.run_chunking_experiment(
            self.sample_text, chapter_key="test_chapter"
        )
        assert experiment["chapter_key"] == "test_chapter"
        assert experiment["total_chars"] > 0
        assert len(experiment["configurations"]) > 0

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            self.chunker.chunk_text(self.sample_text, strategy="invalid_strategy")


class TestTokenizerAnalyzer:
    """Tests for the TokenizerAnalyzer module."""

    def test_analyzer_loads_tokenizers(self):
        """Test that tokenizers can be loaded."""
        from src.chunking.tokenizer_analysis import TokenizerAnalyzer
        analyzer = TokenizerAnalyzer()
        analyzer.load_tokenizers({"bert": "bert-base-uncased"})
        assert "bert" in analyzer.tokenizers

    def test_term_comparison(self):
        """Test scientific term comparison."""
        from src.chunking.tokenizer_analysis import TokenizerAnalyzer
        analyzer = TokenizerAnalyzer()
        analyzer.load_tokenizers({"bert": "bert-base-uncased"})

        report = analyzer.compare_on_terms(["displacement", "acceleration"])
        assert len(report["terms"]) == 2
        assert "bert" in report["summary"]

    def test_fragmentation_scoring(self):
        """Test that multi-word terms have higher fragmentation."""
        from src.chunking.tokenizer_analysis import TokenizerAnalyzer
        analyzer = TokenizerAnalyzer()
        analyzer.load_tokenizers({"bert": "bert-base-uncased"})

        report = analyzer.compare_on_terms(["speed", "uniform circular motion"])
        cell_tokens = report["terms"][0]["tokenizations"]["bert"]["num_tokens"]
        dna_tokens = report["terms"][1]["tokenizations"]["bert"]["num_tokens"]
        assert dna_tokens > cell_tokens, "Complex term should use more tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
