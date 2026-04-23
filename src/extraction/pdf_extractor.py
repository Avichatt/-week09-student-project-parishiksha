
# PariShiksha — PDF Extraction Module

# Extracts raw text from NCERT Science PDFs using multiple backends (PyMuPDF,
# pdfplumber) and logs extraction quality metrics. The dual-backend approach
# lets us cross-validate extraction since NCERT PDFs have inconsistent
# rendering (equations as images, mixed encodings, etc.).


import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    RAW_DATA_DIR,
    EXTRACTED_DATA_DIR,
    TARGET_CHAPTERS,
    NCERT_BASE_URL,
)


class PDFExtractor:
    """
    Extracts text from NCERT PDF files using multiple backends.
    
    Why two backends?
    -----------------
    NCERT PDFs are notoriously messy. PyMuPDF (fitz) is fast and good for
    standard text, but pdfplumber handles tables and complex layouts better.
    By running both, we can cross-check extraction quality and pick the
    better output per page.
    
    Usage:
        extractor = PDFExtractor()
        result = extractor.extract_chapter("chapter_5")
        extractor.save_extraction(result, "chapter_5")
    """

    def __init__(self, raw_dir: Path = RAW_DATA_DIR, output_dir: Path = EXTRACTED_DATA_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # Public API


    def extract_chapter(self, chapter_key: str) -> Dict:
        """
        Extract text from a chapter PDF using both backends.
        
        Parameters
        ----------
        chapter_key : str
            Key from TARGET_CHAPTERS (e.g., "chapter_5")
            
        Returns
        -------
        dict
            {
                "chapter_key": str,
                "metadata": {...},
                "pages_fitz": [{"page_num": int, "text": str}, ...],
                "pages_pdfplumber": [{"page_num": int, "text": str}, ...],
                "extraction_quality": {...}
            }
        """
        chapter_info = TARGET_CHAPTERS.get(chapter_key)
        if not chapter_info:
            raise ValueError(f"Unknown chapter key: {chapter_key}. Available: {list(TARGET_CHAPTERS.keys())}")

        pdf_path = self.raw_dir / chapter_info["pdf_filename"]
        if not pdf_path.exists():
            logger.warning(f"PDF not found at {pdf_path}. Attempting download...")
            self._download_pdf(chapter_info["pdf_filename"])
            if not pdf_path.exists():
                raise FileNotFoundError(
                    f"PDF not found: {pdf_path}\n"
                    f"Please download manually from: {NCERT_BASE_URL}/{chapter_info['pdf_filename']}\n"
                    f"Place it in: {self.raw_dir}"
                )

        logger.info(f"Extracting chapter: {chapter_info['title']} from {pdf_path}")

        # Extract with both backends
        pages_fitz = self._extract_with_fitz(pdf_path, chapter_info.get("page_range"))
        pages_pdfplumber = self._extract_with_pdfplumber(pdf_path, chapter_info.get("page_range"))

        # Compute extraction quality metrics
        quality = self._compute_extraction_quality(pages_fitz, pages_pdfplumber)

        result = {
            "chapter_key": chapter_key,
            "metadata": {
                "title": chapter_info["title"],
                "class": chapter_info["class"],
                "subject": chapter_info["subject"],
                "source_file": chapter_info["pdf_filename"],
                "total_pages_fitz": len(pages_fitz),
                "total_pages_pdfplumber": len(pages_pdfplumber),
            },
            "pages_fitz": pages_fitz,
            "pages_pdfplumber": pages_pdfplumber,
            "extraction_quality": quality,
        }

        logger.info(
            f"Extraction complete. Fitz: {len(pages_fitz)} pages, "
            f"Pdfplumber: {len(pages_pdfplumber)} pages. "
            f"Quality score: {quality['overall_agreement']:.2%}"
        )

        return result

    def save_extraction(self, result: Dict, chapter_key: str) -> Path:
        """Save extraction result as JSON for downstream processing."""
        output_path = self.output_dir / f"{chapter_key}_extracted.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved extraction to {output_path}")
        return output_path

    def extract_all_chapters(self) -> Dict[str, Dict]:
        """Extract all configured chapters."""
        results = {}
        for chapter_key in TARGET_CHAPTERS:
            try:
                result = self.extract_chapter(chapter_key)
                self.save_extraction(result, chapter_key)
                results[chapter_key] = result
            except FileNotFoundError as e:
                logger.error(f"Skipping {chapter_key}: {e}")
        return results


    # Backend: PyMuPDF (fitz)
 

    def _extract_with_fitz(
        self, pdf_path: Path, page_range: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Extract text using PyMuPDF (fitz).
        
        PyMuPDF is fast and handles most standard text well. It preserves
        reading order better than many alternatives. However, it can miss
        text embedded in images and may mangle complex equations.
        """
        pages = []
        try:
            doc = fitz.open(str(pdf_path))
            start_page = page_range[0] - 1 if page_range else 0
            end_page = page_range[1] if page_range else len(doc)

            for page_num in range(start_page, min(end_page, len(doc))):
                page = doc[page_num]
                text = page.get_text("text")

                # Extract text blocks with position info for structure detection
                blocks = page.get_text("dict")["blocks"]
                block_info = []
                for block in blocks:
                    if block["type"] == 0:  # text block
                        for line in block.get("lines", []):
                            line_text = " ".join(
                                span["text"] for span in line.get("spans", [])
                            )
                            if line_text.strip():
                                block_info.append({
                                    "text": line_text.strip(),
                                    "bbox": line["bbox"],
                                    "font_size": line["spans"][0]["size"] if line.get("spans") else 0,
                                })

                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "blocks": block_info,
                    "char_count": len(text),
                    "has_images": any(b["type"] == 1 for b in blocks),
                })
            doc.close()
        except Exception as e:
            logger.error(f"Fitz extraction failed for {pdf_path}: {e}")
            raise

        return pages


    # Backend: pdfplumber
  

    def _extract_with_pdfplumber(
        self, pdf_path: Path, page_range: Optional[Tuple[int, int]] = None
    ) -> List[Dict]:
        """
        Extract text using pdfplumber.
        
        pdfplumber is better at handling tables and complex layouts but
        is slower. It also provides word-level bounding boxes, which
        can be useful for detecting section headers.
        """
        pages = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                start_page = (page_range[0] - 1) if page_range else 0
                end_page = page_range[1] if page_range else len(pdf.pages)

                for page_num in range(start_page, min(end_page, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    text = page.extract_text() or ""

                    # Extract tables if any
                    tables = page.extract_tables() or []

                    pages.append({
                        "page_num": page_num + 1,
                        "text": text,
                        "char_count": len(text),
                        "tables": tables,
                        "has_tables": len(tables) > 0,
                    })
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            raise

        return pages

 
    # Quality Assessment
  

    def _compute_extraction_quality(
        self, pages_fitz: List[Dict], pages_pdfplumber: List[Dict]
    ) -> Dict:
        """
        Compare extraction results from both backends.
        
        This is a simple heuristic: we compare character counts per page
        and flag pages where the two backends disagree significantly.
        Large disagreements often indicate problematic pages (images,
        tables, or encoding issues).
        """
        quality = {
            "per_page": [],
            "pages_with_issues": [],
            "overall_agreement": 0.0,
            "fitz_total_chars": 0,
            "pdfplumber_total_chars": 0,
        }

        max_pages = max(len(pages_fitz), len(pages_pdfplumber))
        agreements = []

        for i in range(max_pages):
            fitz_chars = pages_fitz[i]["char_count"] if i < len(pages_fitz) else 0
            plumber_chars = pages_pdfplumber[i]["char_count"] if i < len(pages_pdfplumber) else 0

            quality["fitz_total_chars"] += fitz_chars
            quality["pdfplumber_total_chars"] += plumber_chars

            # Compute agreement ratio
            max_chars = max(fitz_chars, plumber_chars)
            if max_chars == 0:
                agreement = 1.0
            else:
                agreement = min(fitz_chars, plumber_chars) / max_chars

            page_quality = {
                "page_num": i + 1,
                "fitz_chars": fitz_chars,
                "pdfplumber_chars": plumber_chars,
                "agreement": round(agreement, 3),
            }
            quality["per_page"].append(page_quality)
            agreements.append(agreement)

            # Flag pages with low agreement or potential issues
            if agreement < 0.7:
                quality["pages_with_issues"].append({
                    "page_num": i + 1,
                    "issue": "low_backend_agreement",
                    "agreement": round(agreement, 3),
                })

            # Flag pages with images (likely have formulas/diagrams)
            if i < len(pages_fitz) and pages_fitz[i].get("has_images"):
                quality["pages_with_issues"].append({
                    "page_num": i + 1,
                    "issue": "contains_images",
                })

        quality["overall_agreement"] = sum(agreements) / len(agreements) if agreements else 0.0

        return quality

   
    # PDF Download Helper


    def _download_pdf(self, filename: str) -> None:
        """Attempt to download a PDF from the NCERT website."""
        import urllib.request
        url = f"{NCERT_BASE_URL}/{filename}"
        target = self.raw_dir / filename
        try:
            logger.info(f"Downloading {url} -> {target}")
            urllib.request.urlretrieve(url, str(target))
            logger.info(f"Downloaded successfully: {target}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            logger.info(f"Please download manually from: {url}")



# CLI Entry Point

if __name__ == "__main__":
    extractor = PDFExtractor()
    results = extractor.extract_all_chapters()
    for key, result in results.items():
        print(f"\n{'='*60}")
        print(f"Chapter: {result['metadata']['title']}")
        print(f"Pages extracted (fitz): {result['metadata']['total_pages_fitz']}")
        print(f"Pages extracted (plumber): {result['metadata']['total_pages_pdfplumber']}")
        print(f"Quality agreement: {result['extraction_quality']['overall_agreement']:.2%}")
        if result['extraction_quality']['pages_with_issues']:
            print(f"⚠ Pages with issues: {len(result['extraction_quality']['pages_with_issues'])}")
