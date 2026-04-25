
# PariShiksha — Text Chunking Module

# Implements multiple chunking strategies for NCERT Science content.
# This is where tokenizer analysis meets retrieval design.

# Three strategies:
# 1. Fixed-token chunking: Split at fixed token intervals with overlap
# 2. Sentence-based chunking: Group N sentences per chunk
# 3. Semantic-paragraph chunking: Use section boundaries + paragraph breaks

# The chunk-size experiment (Stretch goal) runs all three at multiple sizes
# and measures how well retrieval performs across configurations.


import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nltk
from loguru import logger
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    CHUNKING_CONFIG,
    PROCESSED_DATA_DIR,
    CHUNKING_OUTPUT_DIR,
)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class TextChunker:
    """
    Chunks cleaned NCERT text using multiple strategies.
    
    Why multiple strategies?
    -----------------------
    - Fixed-token chunks are simple but can split mid-sentence or mid-concept
    - Sentence-based chunks preserve grammatical units but ignore topic structure
    - Semantic chunks respect section boundaries but produce uneven sizes
    
    The right choice depends on your retrieval model and generation model's
    context window. This module lets you experiment with all three and compare.
    
    Usage:
        chunker = TextChunker(tokenizer_name="bert-base-uncased")
        
        # Single strategy
        chunks = chunker.chunk_text(text, strategy="fixed_token", chunk_size=256)
        
        # Run all strategies for comparison
        results = chunker.run_chunking_experiment(text, chapter_key="chapter_5")
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        config: Optional[Dict] = None,
        output_dir: Path = CHUNKING_OUTPUT_DIR,
    ):
        self.config = config or CHUNKING_CONFIG
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load tokenizer for token counting
        logger.info(f"Loading tokenizer for chunking: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name

 
    # Public API


    def chunk_text(
        self,
        text: str,
        strategy: str = "fixed_token",
        chunk_size: int = 256,
        overlap_ratio: float = 0.15,
        sections: Optional[List[Dict]] = None,
        chapter_key: Optional[str] = None,
        source_file: Optional[str] = None,
    ) -> List[Dict]:
        """
        Chunk text with source traceability and stable IDs.
        """
        if strategy == "fixed_token":
            chunks = self._chunk_fixed_token(text, chunk_size, overlap_ratio)
        elif strategy == "sentence_based":
            chunks = self._chunk_sentence_based(text, chunk_size, overlap_ratio)
        elif strategy == "semantic_paragraph":
            if sections is None:
                logger.warning("semantic_paragraph requires sections. Falling back.")
                chunks = self._chunk_paragraph_fallback(text, chunk_size)
            else:
                chunks = self._chunk_semantic_paragraph(sections, chunk_size)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Add industrial metadata & Stable IDs (Gap 1, 2, 3)
        for chunk in chunks:
            # 1. Content-based stable ID (Gap 2)
            chunk["chunk_id"] = hashlib.md5(chunk["text"].encode()).hexdigest()
            
            # 2. Source Traceability (Gap 1)
            chunk["metadata"].update({
                "chapter_key": chapter_key,
                "source_file": source_file,
                "strategy": strategy,
                "chunk_size_config": chunk_size
            })
            
            # 3. Heading Context Injection (Gap 3)
            # If we don't have a heading, we use chapter as context
            if "section_heading" not in chunk["metadata"]:
                context_prefix = f"[{chapter_key.replace('_', ' ').title()}]" if chapter_key else "[NCERT Science]"
                chunk["text"] = f"{context_prefix}\n{chunk['text']}"

        # Filter out tiny chunks
        min_tokens = self.config.get("min_chunk_tokens", 50)
        filtered = [c for c in chunks if c["token_count"] >= min_tokens]
        return filtered

    def run_chunking_experiment(
        self,
        text: str,
        chapter_key: str,
        sections: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Run all chunking strategies at all configured sizes.
        
        This is the core of the Stretch goal: systematically comparing
        chunk strategies to understand their impact on retrieval.
        
        Returns
        -------
        dict
            Full experiment results with statistics per configuration
        """
        logger.info(f"Running chunking experiment for {chapter_key}")
        experiment = {
            "chapter_key": chapter_key,
            "tokenizer": self.tokenizer_name,
            "total_chars": len(text),
            "total_tokens": len(self.tokenizer.tokenize(text)),
            "configurations": [],
        }

        strategies = self.config.get("strategies", ["fixed_token"])
        chunk_sizes = self.config.get("fixed_token_sizes", [256])
        overlap = self.config.get("overlap_ratio", 0.15)

        for strategy in strategies:
            for size in chunk_sizes:
                logger.info(f"  Strategy: {strategy}, Size: {size}")
                chunks = self.chunk_text(
                    text, strategy=strategy, chunk_size=size,
                    overlap_ratio=overlap, sections=sections
                )

                # Compute statistics
                token_counts = [c["token_count"] for c in chunks]
                config_result = {
                    "strategy": strategy,
                    "target_chunk_size": size,
                    "num_chunks": len(chunks),
                    "avg_tokens_per_chunk": round(sum(token_counts) / len(token_counts), 1) if token_counts else 0,
                    "min_tokens": min(token_counts) if token_counts else 0,
                    "max_tokens": max(token_counts) if token_counts else 0,
                    "std_tokens": round(
                        (sum((t - sum(token_counts)/len(token_counts))**2 for t in token_counts) / len(token_counts)) ** 0.5,
                        1
                    ) if token_counts else 0,
                    "chunks": chunks,
                }
                experiment["configurations"].append(config_result)

        # Save experiment
        output_path = self.output_dir / f"{chapter_key}_chunking_experiment.json"
        # Save without the full chunk texts (too large for readable JSON)
        experiment_summary = {
            k: v for k, v in experiment.items() if k != "configurations"
        }
        experiment_summary["configurations"] = [
            {k: v for k, v in cfg.items() if k != "chunks"}
            for cfg in experiment["configurations"]
        ]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, indent=2)
        logger.info(f"Experiment summary saved to {output_path}")

        return experiment

   
    # Strategy : Fixed-Token Chunking
  

    def _chunk_fixed_token(
        self, text: str, chunk_size: int, overlap_ratio: float
    ) -> List[Dict]:
        """
        Split text into fixed-size token chunks with overlap.
        
        This is the simplest strategy. It guarantees uniform chunk sizes
        but can split mid-sentence, mid-concept, or even mid-word (depending
        on tokenizer). The overlap helps maintain context continuity.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        overlap_tokens = int(chunk_size * overlap_ratio)
        stride = chunk_size - overlap_tokens

        chunks = []
        for start in range(0, len(tokens), stride):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            chunks.append({
                "text": chunk_text.strip(),
                "token_count": len(chunk_tokens),
                "start_token": start,
                "end_token": end,
                "metadata": {
                    "overlap_with_previous": overlap_tokens if start > 0 else 0,
                },
            })

            if end >= len(tokens):
                break

        return chunks

  
    # Strategy : Sentence-Based Chunking


    def _chunk_sentence_based(
        self, text: str, target_chunk_size: int, overlap_ratio: float
    ) -> List[Dict]:
        """
        Group sentences until reaching target token count.
        
        This preserves sentence boundaries (important for science text
        where a concept explanation spans exactly one sentence) but
        chunks can be unevenly sized.
        """
        sentences = nltk.sent_tokenize(text)
        target_overlap_tokens = int(target_chunk_size * overlap_ratio)

        chunks = []
        current_sentences = []
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sent_tokens = len(self.tokenizer.tokenize(sentence))

            if current_tokens + sent_tokens > target_chunk_size and current_sentences:
                # Save current chunk
                chunk_text = " ".join(current_sentences)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "num_sentences": len(current_sentences),
                    "metadata": {},
                })

                # Determine overlap based on tokens (Gap 4)
                overlap = []
                overlap_tokens = 0
                for s in reversed(current_sentences):
                    s_tok = len(self.tokenizer.tokenize(s))
                    if overlap_tokens + s_tok <= target_overlap_tokens:
                        overlap.insert(0, s)
                        overlap_tokens += s_tok
                    else:
                        break
                
                current_sentences = overlap
                current_tokens = overlap_tokens

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Last chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text,
                "token_count": current_tokens,
                "num_sentences": len(current_sentences),
                "metadata": {
                    "first_sentence": current_sentences[0][:80],
                },
            })

        return chunks

 
    # Strategy : Semantic Paragraph Chunking
 

    def _chunk_semantic_paragraph(
        self, sections: List[Dict], target_chunk_size: int
    ) -> List[Dict]:
        """
        Chunk based on section structure from TextCleaner.
        
        This respects the textbook's own section boundaries, so a chunk
        about "Cell Wall" won't bleed into "Cell Membrane". Each section
        becomes one or more chunks depending on length.
        
        This is the most semantically meaningful strategy but produces
        the most uneven chunk sizes.
        """
        chunks = []

        for section in sections:
            section_text = section.get("text", "")
            section_heading = section.get("heading", "")
            section_type = section.get("content_type", "narrative")

            if not section_text.strip():
                continue

            section_tokens = len(self.tokenizer.tokenize(section_text))

            if section_tokens <= target_chunk_size:
                # Section fits in one chunk — keep it whole
                chunk_text = f"[{section_heading}]\n{section_text}" if section_heading else section_text
                chunks.append({
                    "text": chunk_text,
                    "token_count": len(self.tokenizer.tokenize(chunk_text)),
                    "metadata": {
                        "section_heading": section_heading,
                        "content_type": section_type,
                        "split": False,
                    },
                })
            else:
                # Section is too long — split by paragraphs
                paragraphs = re.split(r"\n\s*\n", section_text)
                current_paras = []
                current_tokens = 0

                for para in paragraphs:
                    para_tokens = len(self.tokenizer.tokenize(para))

                    if current_tokens + para_tokens > target_chunk_size and current_paras:
                        chunk_text = f"[{section_heading}]\n" + "\n\n".join(current_paras)
                        chunks.append({
                            "text": chunk_text,
                            "token_count": len(self.tokenizer.tokenize(chunk_text)),
                            "metadata": {
                                "section_heading": section_heading,
                                "content_type": section_type,
                                "split": True,
                            },
                        })
                        current_paras = []
                        current_tokens = 0

                    current_paras.append(para)
                    current_tokens += para_tokens

                if current_paras:
                    chunk_text = f"[{section_heading}]\n" + "\n\n".join(current_paras)
                    chunks.append({
                        "text": chunk_text,
                        "token_count": len(self.tokenizer.tokenize(chunk_text)),
                        "metadata": {
                            "section_heading": section_heading,
                            "content_type": section_type,
                            "split": True,
                        },
                    })

        return chunks

    def _chunk_paragraph_fallback(
        self, text: str, target_chunk_size: int
    ) -> List[Dict]:
        """
        Fallback: split by double-newline paragraphs when no section info available.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current_paras = []
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_tokens = len(self.tokenizer.tokenize(para))

            if current_tokens + para_tokens > target_chunk_size and current_paras:
                chunk_text = "\n\n".join(current_paras)
                chunks.append({
                    "text": chunk_text,
                    "token_count": current_tokens,
                    "metadata": {"split_type": "paragraph_fallback"},
                })
                current_paras = []
                current_tokens = 0

            current_paras.append(para)
            current_tokens += para_tokens

        if current_paras:
            chunk_text = "\n\n".join(current_paras)
            chunks.append({
                "text": chunk_text,
                "token_count": current_tokens,
                "metadata": {"split_type": "paragraph_fallback"},
            })

        return chunks


    # Utility


    def save_chunks(self, chunks: List[Dict], chapter_key: str, label: str) -> Path:
        """Save chunks to JSON for downstream retrieval."""
        output_path = self.output_dir / f"{chapter_key}_{label}_chunks.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return output_path

    def print_chunk_stats(self, chunks: List[Dict], label: str = "") -> None:
        """Print summary statistics for a set of chunks."""
        if not chunks:
            print(f"  [{label}] No chunks")
            return

        token_counts = [c["token_count"] for c in chunks]
        print(f"\n  [{label}]")
        print(f"    Chunks: {len(chunks)}")
        print(f"    Avg tokens: {sum(token_counts)/len(token_counts):.0f}")
        print(f"    Min/Max: {min(token_counts)}/{max(token_counts)}")
        print(f"    Total tokens: {sum(token_counts)}")



# CLI Entry Point

if __name__ == "__main__":
    chunker = TextChunker(tokenizer_name="bert-base-uncased")

    # Try to load processed text
    for chapter_key in ["chapter_5", "chapter_6"]:
        text_path = PROCESSED_DATA_DIR / f"{chapter_key}_clean.txt"
        sections_path = PROCESSED_DATA_DIR / f"{chapter_key}_sections.json"

        if not text_path.exists():
            print(f"No processed text for {chapter_key}. Run extraction + cleaning first.")
            continue

        text = text_path.read_text(encoding="utf-8")
        sections = None
        if sections_path.exists():
            with open(sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)

        print(f"\n{'='*60}")
        print(f"CHUNKING EXPERIMENT: {chapter_key}")
        print(f"{'='*60}")

        experiment = chunker.run_chunking_experiment(text, chapter_key, sections)

        for cfg in experiment["configurations"]:
            chunker.print_chunk_stats(
                cfg["chunks"],
                f"{cfg['strategy']} @ {cfg['target_chunk_size']}"
            )
