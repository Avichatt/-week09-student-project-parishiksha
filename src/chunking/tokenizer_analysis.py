
# PariShiksha — Tokenizer Analysis Module

# Compares how different tokenizers handle NCERT Science vocabulary.
# This is Stage 1 of the project: understanding that tokenizer choice
# directly impacts chunking quality and downstream retrieval.
#
# Key insight: scientific terms like "photosynthesis" or "endoplasmic reticulum"
# are split very differently by BPE (GPT-2), WordPiece (BERT), and
# SentencePiece (T5). This affects both token count per chunk and
# semantic integrity of scientific vocabulary in embeddings.


import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (
    TOKENIZER_MODELS,
    SCIENCE_TERMS,
    TOKENIZER_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
)


class TokenizerAnalyzer:
    """
    Compares tokenizer behavior on NCERT Science content.
    
    What this teaches you:
    ---------------------
    1. Different tokenizers fragment scientific terms differently
    2. Token count differs across tokenizers for the same text → chunk sizes vary
    3. Sub-word splits can break semantic meaning (e.g., "mito" + "chond" + "ria")
    4. The tokenizer you choose sets a ceiling on retrieval quality
    
    Usage:
        analyzer = TokenizerAnalyzer()
        analyzer.load_tokenizers()
        term_report = analyzer.compare_on_terms()
        text_report = analyzer.compare_on_text("chapter_5")
        analyzer.generate_comparison_report()
    """

    def __init__(self, output_dir: Path = TOKENIZER_OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.results: Dict = {}

    # -------------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------------

    def load_tokenizers(self, model_names: Optional[Dict[str, str]] = None) -> None:
        """
        Load tokenizers for comparison.
        
        Parameters
        ----------
        model_names : dict, optional
            Mapping of label -> model_name. Defaults to TOKENIZER_MODELS from config.
        """
        models = model_names or TOKENIZER_MODELS
        for label, model_name in models.items():
            try:
                logger.info(f"Loading tokenizer: {label} ({model_name})")
                self.tokenizers[label] = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"Failed to load {label}: {e}")

        logger.info(f"Loaded {len(self.tokenizers)} tokenizers")


    # Analysis: Scientific Terms
 

    def compare_on_terms(
        self, terms: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare how each tokenizer splits scientific terms.
        
        Returns a detailed report showing token IDs, sub-words, and token count
        for each term across all loaded tokenizers.
        """
        terms = terms or SCIENCE_TERMS
        report = {"terms": [], "summary": {}}

        for term in terms:
            term_result = {"term": term, "tokenizations": {}}

            for label, tokenizer in self.tokenizers.items():
                tokens = tokenizer.tokenize(term)
                token_ids = tokenizer.encode(term, add_special_tokens=False)

                term_result["tokenizations"][label] = {
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "num_tokens": len(tokens),
                    # Fragmentation score: 1.0 = single token, higher = more fragmented
                    "fragmentation": len(tokens) / max(len(term.split()), 1),
                }

            report["terms"].append(term_result)

        # Summary statistics per tokenizer
        for label in self.tokenizers:
            counts = [
                t["tokenizations"][label]["num_tokens"]
                for t in report["terms"]
                if label in t["tokenizations"]
            ]
            report["summary"][label] = {
                "mean_tokens_per_term": round(np.mean(counts), 2) if counts else 0,
                "max_tokens_per_term": max(counts) if counts else 0,
                "min_tokens_per_term": min(counts) if counts else 0,
                "total_tokens": sum(counts),
            }

        self.results["term_comparison"] = report
        return report


    # Analysis: Full Chapter Text


    def compare_on_text(
        self, chapter_key: str, text: Optional[str] = None
    ) -> Dict:
        """
        Compare tokenizer behavior on full chapter text.
        
        Measures token count, vocabulary coverage, and how many tokens
        are needed to represent the same content.
        """
        if text is None:
            text_path = PROCESSED_DATA_DIR / f"{chapter_key}_clean.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"Clean text not found: {text_path}")
            text = text_path.read_text(encoding="utf-8")

        report = {"chapter_key": chapter_key, "char_count": len(text), "tokenizers": {}}

        for label, tokenizer in self.tokenizers.items():
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text, add_special_tokens=False)

            # Compute vocabulary overlap
            unique_tokens = set(tokens)

            # Compute compression ratio (chars per token)
            compression = len(text) / len(tokens) if tokens else 0

            report["tokenizers"][label] = {
                "total_tokens": len(tokens),
                "unique_tokens": len(unique_tokens),
                "compression_ratio": round(compression, 2),
                # How many 512-token chunks would this chapter produce?
                "chunks_at_512": len(tokens) // 512 + (1 if len(tokens) % 512 else 0),
                "chunks_at_256": len(tokens) // 256 + (1 if len(tokens) % 256 else 0),
                "chunks_at_128": len(tokens) // 128 + (1 if len(tokens) % 128 else 0),
            }

        self.results["text_comparison"] = report
        return report


    # Visualization


    def plot_term_comparison(self, save: bool = True) -> None:
        """
        Create a bar chart comparing token counts per scientific term
        across tokenizers.
        """
        if "term_comparison" not in self.results:
            logger.warning("Run compare_on_terms() first")
            return

        report = self.results["term_comparison"]
        terms = [t["term"] for t in report["terms"]]
        tokenizer_labels = list(self.tokenizers.keys())

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(terms))
        width = 0.8 / len(tokenizer_labels)
        colors = plt.cm.Set2(np.linspace(0, 1, len(tokenizer_labels)))

        for i, label in enumerate(tokenizer_labels):
            counts = [
                t["tokenizations"].get(label, {}).get("num_tokens", 0)
                for t in report["terms"]
            ]
            ax.bar(x + i * width, counts, width, label=label, color=colors[i])

        ax.set_xlabel("Scientific Term", fontsize=12)
        ax.set_ylabel("Number of Tokens", fontsize=12)
        ax.set_title("Tokenizer Comparison: How Scientific Terms Are Split", fontsize=14)
        ax.set_xticks(x + width * (len(tokenizer_labels) - 1) / 2)
        ax.set_xticklabels(terms, rotation=45, ha="right", fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        if save:
            path = self.output_dir / "tokenizer_term_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {path}")

        plt.show()

    def plot_text_comparison(self, save: bool = True) -> None:
        """
        Create a comparison chart showing token counts and chunk counts
        across tokenizers for full chapter text.
        """
        if "text_comparison" not in self.results:
            logger.warning("Run compare_on_text() first")
            return

        report = self.results["text_comparison"]
        labels = list(report["tokenizers"].keys())
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Token count comparison
        total_tokens = [report["tokenizers"][l]["total_tokens"] for l in labels]
        colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
        axes[0].bar(labels, total_tokens, color=colors)
        axes[0].set_title("Total Tokens for Chapter", fontsize=13)
        axes[0].set_ylabel("Token Count")
        axes[0].tick_params(axis="x", rotation=30)

        # Compression ratio
        compression = [report["tokenizers"][l]["compression_ratio"] for l in labels]
        axes[1].bar(labels, compression, color=colors)
        axes[1].set_title("Compression Ratio (chars/token)", fontsize=13)
        axes[1].set_ylabel("Characters per Token")
        axes[1].tick_params(axis="x", rotation=30)

        # Chunk count at different sizes
        chunk_sizes = [128, 256, 512]
        x = np.arange(len(labels))
        width = 0.25
        for i, size in enumerate(chunk_sizes):
            key = f"chunks_at_{size}"
            counts = [report["tokenizers"][l][key] for l in labels]
            axes[2].bar(x + i * width, counts, width, label=f"{size} tokens")
        axes[2].set_title("Estimated Chunks by Size", fontsize=13)
        axes[2].set_xticks(x + width)
        axes[2].set_xticklabels(labels, rotation=30)
        axes[2].legend()
        axes[2].set_ylabel("Number of Chunks")

        plt.tight_layout()

        if save:
            path = self.output_dir / "tokenizer_text_comparison.png"
            plt.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot: {path}")

        plt.show()

    # Report Generation


    def generate_comparison_report(self) -> Dict:
        """Generate a comprehensive comparison report as JSON."""
        report = {
            "tokenizers_compared": list(self.tokenizers.keys()),
            "term_analysis": self.results.get("term_comparison"),
            "text_analysis": self.results.get("text_comparison"),
            "recommendations": self._generate_recommendations(),
        }

        # Save report
        report_path = self.output_dir / "tokenizer_comparison_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"Saved report: {report_path}")

        return report

    def _generate_recommendations(self) -> Dict:
        """
        Generate recommendations based on analysis results.
        
        Key considerations for NCERT Science content:
        1. Scientific term fragmentation → affects embedding quality
        2. Token economy → affects chunk count and thus retrieval
        3. Hindi/English code-switching readiness
        """
        recommendations = {
            "for_chunking": "",
            "for_embedding": "",
            "rationale": "",
        }

        if "term_comparison" in self.results:
            summary = self.results["term_comparison"]["summary"]
            # Find tokenizer with least fragmentation
            best = min(summary.items(), key=lambda x: x[1]["mean_tokens_per_term"])
            worst = max(summary.items(), key=lambda x: x[1]["mean_tokens_per_term"])

            recommendations["for_chunking"] = (
                f"Prefer '{best[0]}' for chunk size estimation. It uses "
                f"{best[1]['mean_tokens_per_term']:.1f} tokens/term on average vs "
                f"{worst[1]['mean_tokens_per_term']:.1f} for '{worst[0]}'. "
                f"Fewer tokens per scientific term means your chunks preserve more "
                f"semantic content within the same token budget."
            )
            recommendations["for_embedding"] = (
                f"For semantic search, consider that '{worst[0]}' fragments "
                f"scientific terms the most. This can hurt embedding quality for "
                f"science-specific queries. If using SBERT, the tokenizer is fixed "
                f"to the model's, but chunk boundaries should be set considering the "
                f"embedding model's tokenizer."
            )
            recommendations["rationale"] = (
                "Tokenizer choice is a pre-retrieval decision with downstream consequences. "
                "A tokenizer that fragments 'photosynthesis' into 4 sub-words will produce "
                "different chunk boundaries than one that keeps it as 1-2 tokens. This "
                "affects both chunk content density and embedding fidelity."
            )

        return recommendations



# CLI Entry Point

if __name__ == "__main__":
    analyzer = TokenizerAnalyzer()
    analyzer.load_tokenizers()

    # Compare on scientific terms
    print("\n" + "=" * 60)
    print("TOKENIZER COMPARISON: Scientific Terms")
    print("=" * 60)
    term_report = analyzer.compare_on_terms()

    for term_result in term_report["terms"]:
        print(f"\n  Term: '{term_result['term']}'")
        for label, tok_result in term_result["tokenizations"].items():
            print(f"    {label:15s} → {tok_result['num_tokens']:2d} tokens: {tok_result['tokens']}")

    print("\n\nSummary:")
    for label, stats in term_report["summary"].items():
        print(f"  {label:15s}: mean={stats['mean_tokens_per_term']:.1f}, "
              f"max={stats['max_tokens_per_term']}, min={stats['min_tokens_per_term']}")

    # Try to compare on chapter text if available
    try:
        text_report = analyzer.compare_on_text("chapter_5")
        print("\n" + "=" * 60)
        print("TOKENIZER COMPARISON: Full Chapter Text")
        print("=" * 60)
        for label, stats in text_report["tokenizers"].items():
            print(f"\n  {label}:")
            print(f"    Total tokens: {stats['total_tokens']}")
            print(f"    Compression: {stats['compression_ratio']} chars/token")
            print(f"    Chunks @512: {stats['chunks_at_512']}")
    except FileNotFoundError:
        print("\nSkipping text comparison (no processed chapter text found)")

    # Generate plots
    analyzer.plot_term_comparison()

    # Generate report
    report = analyzer.generate_comparison_report()
    print(f"\nReport saved to: {analyzer.output_dir / 'tokenizer_comparison_report.json'}")
