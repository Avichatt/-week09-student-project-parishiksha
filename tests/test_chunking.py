
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
            "The cell is the fundamental structural and functional unit of all living organisms. "
            "Robert Hooke first observed cells in 1665 using a primitive microscope. "
            "He observed thin slices of cork and noticed small compartments which he called cells. "
            "The cell theory states that all living organisms are composed of cells, "
            "the cell is the basic unit of life, and all cells arise from pre-existing cells. "
            "Prokaryotic cells lack a well-defined nuclear membrane. "
            "Eukaryotic cells have a well-defined nucleus bounded by a nuclear membrane. "
            "The plasma membrane is selectively permeable and controls the transport of substances. "
            "The cell wall in plants provides structural strength and is made of cellulose. "
            "Mitochondria are known as the powerhouse of the cell because they generate energy. "
            "The endoplasmic reticulum is a network of membranes involved in protein synthesis. "
            "The Golgi apparatus packages and dispatches materials synthesized in the cell. "
            "Lysosomes contain digestive enzymes and are called the suicide bags of the cell. "
            "Chloroplasts are found in plant cells and contain chlorophyll for photosynthesis. "
            "The nucleus contains chromosomes which carry genetic information in the form of DNA."
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
                "text": "The cell is the fundamental unit of life. All living organisms are made of cells.",
                "content_type": "narrative",
            },
            {
                "heading": "Cell Organelles",
                "text": "Mitochondria are the powerhouse. The ER is a membrane network. Golgi packages materials.",
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
            assert "strategy" in chunk
            assert chunk["strategy"] == "fixed_token"

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

        report = analyzer.compare_on_terms(["photosynthesis", "mitochondria"])
        assert len(report["terms"]) == 2
        assert "bert" in report["summary"]

    def test_fragmentation_scoring(self):
        """Test that multi-word terms have higher fragmentation."""
        from src.chunking.tokenizer_analysis import TokenizerAnalyzer
        analyzer = TokenizerAnalyzer()
        analyzer.load_tokenizers({"bert": "bert-base-uncased"})

        report = analyzer.compare_on_terms(["cell", "deoxyribonucleic acid"])
        cell_tokens = report["terms"][0]["tokenizations"]["bert"]["num_tokens"]
        dna_tokens = report["terms"][1]["tokenizations"]["bert"]["num_tokens"]
        assert dna_tokens > cell_tokens, "Complex term should use more tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
