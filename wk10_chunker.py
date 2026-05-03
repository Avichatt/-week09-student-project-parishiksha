# =============================================================================
# PariShiksha Wk10 — Stage 1: Content-Type-Aware Chunker
# =============================================================================
# Implements the Wk10 rubric requirements:
#   - Content-type metadata: prose / worked_example / question_or_exercise
#   - Token-aware sizing with tiktoken (~250 tokens)
#   - Worked-example tables preserved
#   - Chunk metadata: {source, section, content_type, page}
#   - Persist as wk10_chunks.json
# =============================================================================

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken
from loguru import logger


class Wk10Chunker:
    """
    Content-type-aware chunker for NCERT Science text.
    
    Three content types detected via regex:
      - prose: narrative textbook content
      - worked_example: Examples with solutions (Example 4.1, etc.)
      - question_or_exercise: Activities, Pause and Ponder, exercises
    
    Uses tiktoken (cl100k_base) for token counting per rubric requirement.
    Target chunk size: ~250 tokens.
    """

    def __init__(self, max_tokens: int = 250):
        self.max_tokens = max_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")
        
        # Regex patterns for content type detection
        self.example_pattern = re.compile(
            r"^Example\s+\d+\.\d+:", re.MULTILINE
        )
        self.activity_pattern = re.compile(
            r"^Activity\s+\d+\.\d+:", re.MULTILINE
        )
        self.exercise_patterns = [
            re.compile(r"^Pause and Ponder", re.MULTILINE),
            re.compile(r"^Revise, Reflect, Refine", re.MULTILINE),
            re.compile(r"^The Journey Beyond", re.MULTILINE),
            re.compile(r"^\d+\.\s+(?:A |The |Find |What |How |Is |Calculate |Consider )", re.MULTILINE),
        ]
        self.table_pattern = re.compile(r"Table\s+\d+\.\d+:")

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken cl100k_base."""
        return len(self.enc.encode(text))

    def classify_content_type(self, text: str, heading: str = "") -> str:
        """
        Classify a text block into one of three content types.
        
        Returns: 'prose', 'worked_example', or 'question_or_exercise'
        """
        # Check for worked examples first (highest priority)
        if self.example_pattern.search(text):
            return "worked_example"
        if "Answer:" in text and ("Example" in heading or "Example" in text[:50]):
            return "worked_example"
        
        # Check for activities and exercises
        if self.activity_pattern.search(text):
            return "question_or_exercise"
        for pat in self.exercise_patterns:
            if pat.search(text):
                return "question_or_exercise"
        if any(kw in heading for kw in ["Pause and Ponder", "Revise", "Journey Beyond", "Activity"]):
            return "question_or_exercise"
        
        # Default: prose
        return "prose"

    def chunk_sections(
        self,
        sections: List[Dict],
        source_file: str = "iesc104.pdf",
        chapter_key: str = "chapter_4",
    ) -> List[Dict]:
        """
        Chunk structured sections with content-type awareness.
        
        Each chunk gets metadata: {source, section, content_type, page, chunk_id}
        Worked examples and tables are preserved as complete units when possible.
        """
        all_chunks = []
        chunk_index = 0

        for section in sections:
            heading = section.get("heading", "Unknown")
            text = section.get("text", "")
            section_content_type = section.get("content_type", "narrative")

            if not text.strip():
                continue

            # Split section into logical blocks
            blocks = self._split_into_blocks(text, heading)

            for block_text, block_type in blocks:
                block_tokens = self.count_tokens(block_text)

                if block_tokens <= self.max_tokens:
                    # Block fits in one chunk — keep it whole
                    chunk = self._make_chunk(
                        text=block_text,
                        section=heading,
                        content_type=block_type,
                        source=source_file,
                        chapter=chapter_key,
                        index=chunk_index,
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1
                else:
                    # Block too large — split by sentences, respecting type
                    sub_chunks = self._split_large_block(
                        block_text, block_type, heading, source_file, 
                        chapter_key, chunk_index
                    )
                    all_chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)

        logger.info(
            f"Chunked {len(sections)} sections into {len(all_chunks)} chunks "
            f"(target: {self.max_tokens} tokens)"
        )
        
        # Log content type distribution
        type_counts = {}
        for c in all_chunks:
            ct = c["metadata"]["content_type"]
            type_counts[ct] = type_counts.get(ct, 0) + 1
        logger.info(f"Content type distribution: {type_counts}")
        
        return all_chunks

    def _split_into_blocks(self, text: str, heading: str) -> List[tuple]:
        """
        Split section text into logical blocks, preserving worked examples
        and exercises as complete units.
        
        Returns list of (block_text, content_type) tuples.
        """
        blocks = []
        
        # Try to split on Example markers first
        example_splits = re.split(
            r"(Example\s+\d+\.\d+:.*?)(?=Example\s+\d+\.\d+:|Activity\s+\d+\.\d+:|Pause and Ponder|$)",
            text,
            flags=re.DOTALL
        )
        
        # Also split on Activity markers
        parts = []
        for part in example_splits:
            if not part.strip():
                continue
            # Further split on Activity boundaries
            activity_splits = re.split(
                r"(Activity\s+\d+\.\d+:.*?)(?=Example\s+\d+\.\d+:|Activity\s+\d+\.\d+:|$)",
                part,
                flags=re.DOTALL
            )
            parts.extend([p for p in activity_splits if p.strip()])
        
        if not parts:
            parts = [text]
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            content_type = self.classify_content_type(part, heading)
            blocks.append((part, content_type))
        
        return blocks

    def _split_large_block(
        self, text: str, content_type: str, heading: str,
        source: str, chapter: str, start_index: int
    ) -> List[Dict]:
        """Split a large block into chunks of ~max_tokens, respecting sentence boundaries."""
        import nltk
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_sents = []
        current_tokens = 0
        idx = start_index

        for sent in sentences:
            sent_tokens = self.count_tokens(sent)
            
            if current_tokens + sent_tokens > self.max_tokens and current_sents:
                chunk_text = " ".join(current_sents)
                chunk = self._make_chunk(
                    text=chunk_text,
                    section=heading,
                    content_type=content_type,
                    source=source,
                    chapter=chapter,
                    index=idx,
                )
                chunks.append(chunk)
                idx += 1
                current_sents = []
                current_tokens = 0

            current_sents.append(sent)
            current_tokens += sent_tokens

        # Last chunk
        if current_sents:
            chunk_text = " ".join(current_sents)
            chunk = self._make_chunk(
                text=chunk_text,
                section=heading,
                content_type=content_type,
                source=source,
                chapter=chapter,
                index=idx,
            )
            chunks.append(chunk)

        return chunks

    def _make_chunk(
        self, text: str, section: str, content_type: str,
        source: str, chapter: str, index: int,
    ) -> Dict:
        """Create a chunk dict with full metadata."""
        # Inject section heading for retrieval context
        display_text = f"[{section}]\n{text}" if section else text
        
        chunk_id = hashlib.md5(
            f"{chapter}:{section}:{index}:{text[:100]}".encode()
        ).hexdigest()[:12]

        return {
            "chunk_id": chunk_id,
            "text": display_text,
            "token_count": self.count_tokens(display_text),
            "metadata": {
                "source": source,
                "section": section,
                "content_type": content_type,
                "page": self._estimate_page(index),
                "chapter": chapter,
                "chunk_index": index,
            },
        }

    def _estimate_page(self, chunk_index: int) -> int:
        """Rough page estimate based on chunk position."""
        # Chapter 4 spans roughly pages 49-72 in the PDF
        return 49 + (chunk_index // 4)

    def save_chunks(self, chunks: List[Dict], output_path: str = "wk10_chunks.json"):
        """Persist chunks as JSON."""
        path = Path(output_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {path}")
        return path


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    
    sections_path = Path("data/processed/chapter_4_sections.json")
    if not sections_path.exists():
        print("No sections file found. Run extraction first.")
        exit(1)
    
    with open(sections_path, "r", encoding="utf-8") as f:
        sections = json.load(f)
    
    chunker = Wk10Chunker(max_tokens=250)
    chunks = chunker.chunk_sections(sections)
    chunker.save_chunks(chunks)
    
    # Print stats
    token_counts = [c["token_count"] for c in chunks]
    types = {}
    for c in chunks:
        t = c["metadata"]["content_type"]
        types[t] = types.get(t, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"WK10 CHUNKING RESULTS")
    print(f"{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg tokens:   {sum(token_counts)/len(token_counts):.0f}")
    print(f"Min/Max:      {min(token_counts)}/{max(token_counts)}")
    print(f"\nContent types: {types}")
    
    for i, c in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i} [{c['metadata']['content_type']}] ({c['token_count']} tokens) ---")
        print(c["text"][:200] + "...")
