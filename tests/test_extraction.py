# =============================================================================
# PariShiksha — Tests for Extraction Module
# =============================================================================

import json
import re
import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.extraction.text_cleaner import TextCleaner


class TestTextCleaner:
    """Tests for the TextCleaner module."""

    def setup_method(self):
        self.cleaner = TextCleaner()

    def test_mojibake_fixing(self):
        """Test that common mojibake patterns are fixed."""
        text = "The temperature is 25â€™C and the equation is H₂O â†' H⁺ + OH⁻"
        cleaned, report = self.cleaner._clean_page_text(text, page_num=1)
        assert "â€™" not in cleaned
        assert report["mojibake_fixes"] > 0

    def test_header_removal(self):
        """Test that NCERT headers/footers are removed."""
        text = "SCIENCE\n42\nThis is the actual content.\n© NCERT"
        cleaned, report = self.cleaner._clean_page_text(text, page_num=1)
        assert "actual content" in cleaned
        assert report["headers_removed"] >= 1

    def test_dangling_reference_detection(self):
        """Test detection of figure references."""
        text = "As shown in Fig. 5.3, the cell membrane is selectively permeable."
        cleaned, report = self.cleaner._clean_page_text(text, page_num=1)
        assert len(report["dangling_refs"]) > 0

    def test_page_merge_hyphenation(self):
        """Test that hyphenated words across pages are rejoined."""
        pages = [
            (1, "The process of photo-"),
            (2, "synthesis is essential for life.")
        ]
        merged = self.cleaner._merge_pages(pages)
        assert "photosynthesis" in merged
        assert "photo-\nsynthesis" not in merged

    def test_section_detection(self):
        """Test that section headings are detected."""
        text = """Introduction paragraph here.

5.1 CELL ORGANELLES

The cell has many organelles including mitochondria.

5.2 CELL MEMBRANE

The cell membrane controls substances."""
        sections = self.cleaner._detect_sections(text)
        assert len(sections) >= 2
        headings = [s["heading"] for s in sections]
        assert any("CELL" in h for h in headings)

    def test_content_type_classification(self):
        """Test content type detection."""
        assert self.cleaner._classify_content("Q. What is a cell?") == "question"
        assert self.cleaner._classify_content("Activity 5.1: Observe the slide") == "activity"
        assert self.cleaner._classify_content("Cells are the basic units.") == "narrative"

    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully."""
        assert self.cleaner._classify_content("") == "empty"

    def test_whitespace_normalization(self):
        """Test that excessive whitespace is normalized."""
        text = "Too   many   spaces   here.\n\n\n\n\nToo many newlines."
        cleaned, _ = self.cleaner._clean_page_text(text, page_num=1)
        assert "   " not in cleaned
        assert "\n\n\n" not in cleaned


class TestExtractionIntegration:
    """Integration tests that require PDF files to be present."""

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent / "data" / "raw").glob("*.pdf"),
        reason="No PDF files available for integration testing"
    )
    def test_full_extraction_pipeline(self):
        """Test the full extraction → cleaning pipeline."""
        from src.extraction.pdf_extractor import PDFExtractor

        extractor = PDFExtractor()
        cleaner = TextCleaner()

        # Map filename to chapter key based on config
        from config.config import TARGET_CHAPTERS
        name_to_key = {v["pdf_filename"]: k for k, v in TARGET_CHAPTERS.items()}

        # This will only run if PDFs are present
        for pdf_file in (Path(__file__).resolve().parent.parent / "data" / "raw").glob("*.pdf"):
            chapter_key = name_to_key.get(pdf_file.name)
            if not chapter_key:
                continue
            
            result = extractor.extract_chapter(chapter_key)
            structured = cleaner.clean_chapter(result)

            assert structured["full_text"]
            assert len(structured["sections"]) > 0
            break  # Just test the first PDF


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
