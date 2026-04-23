# =============================================================================
# PariShiksha — Text Cleaning & Structuring Module
# =============================================================================
# Takes raw extracted text from PDFs and cleans it into structured sections.
# NCERT PDFs are messy: mojibake from formula rendering, dangling figure
# references, inconsistent whitespace, headers/footers repeated on every page.
# This module handles all of that.
# =============================================================================

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import EXTRACTED_DATA_DIR, PROCESSED_DATA_DIR


class TextCleaner:
    """
    Cleans and structures raw PDF-extracted text from NCERT chapters.
    
    The cleaning pipeline:
    1. Remove headers/footers (page numbers, chapter titles repeated on each page)
    2. Fix encoding issues (mojibake from formula rendering)
    3. Normalize whitespace and line breaks
    4. Detect and tag section boundaries (headings, subheadings)
    5. Classify content type (narrative, worked_example, question, activity)
    6. Merge text across page breaks
    7. Tag dangling figure references
    
    Usage:
        cleaner = TextCleaner()
        raw = json.load(open("data/extracted/chapter_5_extracted.json"))
        structured = cleaner.clean_chapter(raw)
        cleaner.save_structured(structured, "chapter_5")
    """

    # Common NCERT header/footer patterns
    HEADER_PATTERNS = [
        r"^\d+\s*$",                                    # bare page numbers
        r"^SCIENCE\s*$",                                # subject header
        r"^S\s*C\s*I\s*E\s*N\s*C\s*E\s*$",            # spaced-out SCIENCE
        r"^\d+\s+SCIENCE\s*$",                          # page number + SCIENCE
        r"^SCIENCE\s+\d+\s*$",                          # SCIENCE + page number
        r"^Chapter\s+\d+.*$",                           # chapter header
        r"^\s*\d{4}-\d{2}\s*$",                         # year-month stamps
        r"^©\s*NCERT.*$",                               # copyright lines
        r"^not to be republished.*$",                    # copyright notice
        r"^Free Distribution.*$",                        # distribution notice
    ]

    # Patterns indicating section headings in NCERT textbooks
    HEADING_PATTERNS = [
        # Numbered sections like "5.1", "5.2.1"
        (r"^\d+\.\d+(?:\.\d+)?\s+[A-Z]", "section_heading"),
        # All-caps headings
        (r"^[A-Z][A-Z\s]{5,}$", "section_heading"),
        # Question section markers
        (r"^(?:Questions?|Exercises?|Activities?)\s*$", "section_heading"),
        # "What you have learnt" type summaries
        (r"^What\s+(?:you\s+have|we\s+have)\s+learnt", "summary_heading"),
    ]

    # Content type detection patterns
    CONTENT_TYPE_PATTERNS = [
        (r"^(?:Q\.|Question|\d+\.)\s+", "question"),
        (r"^(?:Activity|ACTIVITY)\s+\d+", "activity"),
        (r"^(?:Example|EXAMPLE)\s+\d+", "worked_example"),
        (r"^(?:Think about it|More to know|Points to ponder)", "sidebar"),
        (r"^(?:Fig\.|Figure|Fig)\s+\d+", "figure_reference"),
        (r"^(?:Table)\s+\d+", "table_reference"),
    ]

    # Common mojibake patterns from NCERT PDFs
    MOJIBAKE_FIXES = {
        "â€™": "'",
        "â€œ": '"',
        "â€\x9d": '"',
        "â€\x94": "—",
        "â€\x93": "–",
        "Â°": "°",
        "Âµ": "µ",
        "â†'": "→",
        "â‰ˆ": "≈",
        "\ufeff": "",           # BOM
        "\u200b": "",           # zero-width space
        "\u00ad": "",           # soft hyphen
    }

    def __init__(
        self,
        extracted_dir: Path = EXTRACTED_DATA_DIR,
        output_dir: Path = PROCESSED_DATA_DIR,
    ):
        self.extracted_dir = extracted_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compile regex patterns for performance
        self._header_re = [re.compile(p, re.IGNORECASE) for p in self.HEADER_PATTERNS]
        self._heading_re = [(re.compile(p), label) for p, label in self.HEADING_PATTERNS]
        self._content_re = [(re.compile(p, re.IGNORECASE), label) for p, label in self.CONTENT_TYPE_PATTERNS]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def clean_chapter(self, extraction_result: Dict, prefer_backend: str = "fitz") -> Dict:
        """
        Clean and structure an extracted chapter.
        
        Parameters
        ----------
        extraction_result : dict
            Output from PDFExtractor.extract_chapter()
        prefer_backend : str
            Which extraction backend to prefer ("fitz" or "pdfplumber").
            Falls back to the other if the preferred one has too many issues.
            
        Returns
        -------
        dict
            {
                "chapter_key": str,
                "metadata": {...},
                "full_text": str,           # cleaned, merged full text
                "sections": [...],          # structured sections
                "cleaning_report": {...},   # what was cleaned and why
            }
        """
        chapter_key = extraction_result["chapter_key"]
        logger.info(f"Cleaning chapter: {chapter_key}")

        # Select best backend pages
        pages_key = f"pages_{prefer_backend}"
        pages = extraction_result.get(pages_key, [])
        if not pages:
            alt_backend = "pdfplumber" if prefer_backend == "fitz" else "fitz"
            logger.warning(f"No pages from {prefer_backend}, falling back to {alt_backend}")
            pages = extraction_result.get(f"pages_{alt_backend}", [])

        cleaning_report = {
            "backend_used": prefer_backend,
            "total_pages": len(pages),
            "headers_removed": 0,
            "mojibake_fixes": 0,
            "dangling_references": [],
            "empty_pages": [],
            "encoding_issues": [],
        }

        # Step 1: Extract and concatenate text from pages
        page_texts = []
        for page in pages:
            text = page.get("text", "")
            if not text.strip():
                cleaning_report["empty_pages"].append(page.get("page_num", "?"))
                continue
            page_texts.append((page.get("page_num", 0), text))

        # Step 2: Clean each page
        cleaned_pages = []
        for page_num, text in page_texts:
            cleaned, page_report = self._clean_page_text(text, page_num)
            cleaning_report["headers_removed"] += page_report["headers_removed"]
            cleaning_report["mojibake_fixes"] += page_report["mojibake_fixes"]
            if page_report.get("dangling_refs"):
                cleaning_report["dangling_references"].extend(page_report["dangling_refs"])
            cleaned_pages.append((page_num, cleaned))

        # Step 3: Merge text across pages (handle hyphenation at page breaks)
        full_text = self._merge_pages(cleaned_pages)

        # Step 4: Split into structured sections
        sections = self._detect_sections(full_text)

        # Step 5: Classify content types within sections
        for section in sections:
            section["content_type"] = self._classify_content(section["text"])

        result = {
            "chapter_key": chapter_key,
            "metadata": extraction_result.get("metadata", {}),
            "full_text": full_text,
            "sections": sections,
            "cleaning_report": cleaning_report,
        }

        logger.info(
            f"Cleaning complete. {len(sections)} sections found. "
            f"Headers removed: {cleaning_report['headers_removed']}, "
            f"Mojibake fixes: {cleaning_report['mojibake_fixes']}, "
            f"Dangling refs: {len(cleaning_report['dangling_references'])}"
        )

        return result

    def save_structured(self, structured_result: Dict, chapter_key: str) -> Tuple[Path, Path]:
        """
        Save structured result as JSON and also as plain text for chunking.
        
        Returns paths to (json_file, text_file).
        """
        # Save full structured JSON
        json_path = self.output_dir / f"{chapter_key}_structured.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(structured_result, f, ensure_ascii=False, indent=2)

        # Save clean text (for direct chunking input)
        text_path = self.output_dir / f"{chapter_key}_clean.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(structured_result["full_text"])

        # Save sections separately for section-aware chunking
        sections_path = self.output_dir / f"{chapter_key}_sections.json"
        with open(sections_path, "w", encoding="utf-8") as f:
            json.dump(structured_result["sections"], f, ensure_ascii=False, indent=2)

        logger.info(f"Saved: {json_path}, {text_path}, {sections_path}")
        return json_path, text_path

    # -------------------------------------------------------------------------
    # Cleaning Pipeline Steps
    # -------------------------------------------------------------------------

    def _clean_page_text(self, text: str, page_num: int) -> Tuple[str, Dict]:
        """Apply all cleaning steps to a single page's text."""
        report = {"headers_removed": 0, "mojibake_fixes": 0, "dangling_refs": []}

        # Fix mojibake
        for bad, good in self.MOJIBAKE_FIXES.items():
            count = text.count(bad)
            if count > 0:
                text = text.replace(bad, good)
                report["mojibake_fixes"] += count

        # Remove headers and footers
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if any(pattern.match(stripped) for pattern in self._header_re):
                report["headers_removed"] += 1
                continue
            cleaned_lines.append(line)

        # Detect dangling figure references
        for line in cleaned_lines:
            if re.search(r"(?:Fig\.|Figure|Fig)\s+\d+\.\d+", line, re.IGNORECASE):
                report["dangling_refs"].append({
                    "page": page_num,
                    "reference": line.strip()[:100],
                })

        # Normalize whitespace
        text = "\n".join(cleaned_lines)
        text = re.sub(r" {2,}", " ", text)           # multiple spaces -> single
        text = re.sub(r"\n{3,}", "\n\n", text)       # excessive newlines
        text = re.sub(r"\t", " ", text)               # tabs -> spaces

        return text, report

    def _merge_pages(self, cleaned_pages: List[Tuple[int, str]]) -> str:
        """
        Merge page texts, handling hyphenation at page boundaries.
        
        NCERT PDFs often hyphenate words at page breaks (e.g., "photo-\nsynthesis").
        We detect and rejoin these.
        """
        if not cleaned_pages:
            return ""

        merged = []
        for i, (page_num, text) in enumerate(cleaned_pages):
            if i > 0 and merged:
                # Check if previous page ended with a hyphenated word
                last_text = merged[-1]
                if last_text.rstrip().endswith("-"):
                    # Rejoin hyphenated word across page break
                    merged[-1] = last_text.rstrip()[:-1]  # remove hyphen
                    text = text.lstrip()  # remove leading whitespace from next page
                else:
                    merged.append("\n\n")  # page separator

            merged.append(text)

        return "".join(merged)

    def _detect_sections(self, text: str) -> List[Dict]:
        """
        Split text into sections based on heading detection.
        
        Returns a list of dicts, each with:
        - "heading": str (section heading text)
        - "heading_type": str (section_heading, summary_heading, etc.)
        - "text": str (section body text)
        - "start_char": int (position in full text)
        """
        lines = text.split("\n")
        sections = []
        current_section = {
            "heading": "Introduction",
            "heading_type": "implicit",
            "text_lines": [],
            "start_char": 0,
        }

        char_pos = 0
        for line in lines:
            is_heading = False
            heading_type = None

            stripped = line.strip()
            if stripped:
                for pattern, htype in self._heading_re:
                    if pattern.match(stripped):
                        is_heading = True
                        heading_type = htype
                        break

            if is_heading and current_section["text_lines"]:
                # Save current section
                sections.append({
                    "heading": current_section["heading"],
                    "heading_type": current_section["heading_type"],
                    "text": "\n".join(current_section["text_lines"]).strip(),
                    "start_char": current_section["start_char"],
                })
                # Start new section
                current_section = {
                    "heading": stripped,
                    "heading_type": heading_type,
                    "text_lines": [],
                    "start_char": char_pos,
                }
            else:
                current_section["text_lines"].append(line)

            char_pos += len(line) + 1  # +1 for newline

        # Don't forget the last section
        if current_section["text_lines"]:
            sections.append({
                "heading": current_section["heading"],
                "heading_type": current_section["heading_type"],
                "text": "\n".join(current_section["text_lines"]).strip(),
                "start_char": current_section["start_char"],
            })

        return sections

    def _classify_content(self, text: str) -> str:
        """
        Classify a section's content type.
        
        Returns one of: narrative, question, activity, worked_example,
        sidebar, figure_reference, table_reference, mixed.
        """
        if not text.strip():
            return "empty"

        detected_types = set()
        for line in text.split("\n")[:10]:  # check first 10 lines
            stripped = line.strip()
            if not stripped:
                continue
            for pattern, label in self._content_re:
                if pattern.match(stripped):
                    detected_types.add(label)

        if not detected_types:
            return "narrative"
        if len(detected_types) == 1:
            return detected_types.pop()
        return "mixed"


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    import sys

    cleaner = TextCleaner()

    # Process all extracted JSON files
    for json_file in EXTRACTED_DATA_DIR.glob("*_extracted.json"):
        chapter_key = json_file.stem.replace("_extracted", "")
        logger.info(f"Processing: {chapter_key}")

        with open(json_file, "r", encoding="utf-8") as f:
            extraction_result = json.load(f)

        structured = cleaner.clean_chapter(extraction_result)
        cleaner.save_structured(structured, chapter_key)

        print(f"\n{'='*60}")
        print(f"Chapter: {structured['metadata'].get('title', chapter_key)}")
        print(f"Sections found: {len(structured['sections'])}")
        for s in structured["sections"]:
            print(f"  [{s['content_type']}] {s['heading'][:60]}")
        print(f"Cleaning report:")
        report = structured["cleaning_report"]
        print(f"  Headers removed: {report['headers_removed']}")
        print(f"  Mojibake fixes: {report['mojibake_fixes']}")
        print(f"  Dangling refs: {len(report['dangling_references'])}")
        print(f"  Empty pages: {report['empty_pages']}")
