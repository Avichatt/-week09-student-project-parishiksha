# =============================================================================
# PariShiksha — Main Orchestrator
# =============================================================================
# Runs the full pipeline end-to-end:
#   Stage 1: PDF Extraction + Cleaning + Tokenizer Comparison
#   Stage 2: Chunking Experiment (multiple strategies × sizes)
#   Stage 3: Embedding + Retrieval + Generation
#   Stage 4: Evaluation
#
# This script is the single entry point. You can also run each stage
# independently via the module CLI entry points or the notebooks.
#
# Usage:
#   python main.py --stage all         # Run everything
#   python main.py --stage extract     # Stage 1 only
#   python main.py --stage chunk       # Stage 2 only
#   python main.py --stage retrieve    # Stage 3 only
#   python main.py --stage evaluate    # Stage 4 only
# =============================================================================

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("parishiksha.log", level="DEBUG", rotation="10 MB")

from config.config import (
    RAW_DATA_DIR,
    EXTRACTED_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHUNKING_OUTPUT_DIR,
    RETRIEVAL_OUTPUT_DIR,
    EVAL_OUTPUT_DIR,
    TARGET_CHAPTERS,
)


def stage_1_extract_and_clean():
    """
    Stage 1: PDF Extraction, Text Cleaning, and Tokenizer Comparison
    
    This stage:
    1. Downloads NCERT PDFs (if not present)
    2. Extracts text using dual backends (PyMuPDF + pdfplumber)
    3. Compares extraction quality between backends
    4. Cleans text: removes headers, fixes mojibake, normalizes whitespace
    5. Detects section boundaries and content types
    6. Compares tokenizer behavior on scientific vocabulary
    """
    logger.info("=" * 60)
    logger.info("STAGE 1: Extraction, Cleaning & Tokenizer Analysis")
    logger.info("=" * 60)

    # --- Step 1a: Extract PDFs ---
    from src.extraction.pdf_extractor import PDFExtractor
    extractor = PDFExtractor()

    extraction_results = {}
    for chapter_key in TARGET_CHAPTERS:
        try:
            result = extractor.extract_chapter(chapter_key)
            extractor.save_extraction(result, chapter_key)
            extraction_results[chapter_key] = result
            logger.info(
                f"✓ Extracted {chapter_key}: "
                f"{result['metadata']['total_pages_fitz']} pages, "
                f"quality={result['extraction_quality']['overall_agreement']:.0%}"
            )
        except FileNotFoundError as e:
            logger.error(f"✗ {chapter_key}: {e}")
            logger.info("Download PDFs manually and place in data/raw/")

    if not extraction_results:
        logger.error("No chapters extracted. Cannot continue.")
        return False

    # --- Step 1b: Clean text ---
    from src.extraction.text_cleaner import TextCleaner
    cleaner = TextCleaner()

    for chapter_key, extraction_result in extraction_results.items():
        structured = cleaner.clean_chapter(extraction_result)
        cleaner.save_structured(structured, chapter_key)
        report = structured["cleaning_report"]
        logger.info(
            f"✓ Cleaned {chapter_key}: "
            f"{len(structured['sections'])} sections, "
            f"headers_removed={report['headers_removed']}, "
            f"mojibake_fixes={report['mojibake_fixes']}, "
            f"dangling_refs={len(report['dangling_references'])}"
        )

    # --- Step 1c: Tokenizer comparison ---
    from src.chunking.tokenizer_analysis import TokenizerAnalyzer
    analyzer = TokenizerAnalyzer()
    analyzer.load_tokenizers()

    # Compare on scientific terms
    term_report = analyzer.compare_on_terms()
    logger.info("Tokenizer comparison on scientific terms:")
    for label, stats in term_report["summary"].items():
        logger.info(f"  {label}: avg={stats['mean_tokens_per_term']:.1f} tokens/term")

    # Compare on full text (if available)
    for chapter_key in extraction_results:
        try:
            text_report = analyzer.compare_on_text(chapter_key)
            logger.info(f"\nTokenizer comparison on {chapter_key} text:")
            for label, stats in text_report["tokenizers"].items():
                logger.info(
                    f"  {label}: {stats['total_tokens']} tokens, "
                    f"compression={stats['compression_ratio']:.1f} chars/token, "
                    f"chunks@256={stats['chunks_at_256']}"
                )
        except FileNotFoundError:
            pass

    # Generate plots and report
    try:
        analyzer.plot_term_comparison()
        analyzer.plot_text_comparison()
    except Exception as e:
        logger.warning(f"Plot generation failed (non-critical): {e}")

    analyzer.generate_comparison_report()
    logger.info("✓ Stage 1 complete")
    return True


def stage_2_chunking():
    """
    Stage 2: Chunking Experiment
    
    Runs all three chunking strategies at multiple sizes:
    - fixed_token @ [128, 256, 512] tokens
    - sentence_based @ [128, 256, 512] tokens
    - semantic_paragraph @ [128, 256, 512] tokens
    
    Saves all configurations for retrieval comparison in Stage 3.
    """
    logger.info("=" * 60)
    logger.info("STAGE 2: Chunking Experiment")
    logger.info("=" * 60)

    from src.chunking.chunker import TextChunker
    chunker = TextChunker(tokenizer_name="bert-base-uncased")

    for chapter_key in TARGET_CHAPTERS:
        text_path = PROCESSED_DATA_DIR / f"{chapter_key}_clean.txt"
        sections_path = PROCESSED_DATA_DIR / f"{chapter_key}_sections.json"

        if not text_path.exists():
            logger.warning(f"No processed text for {chapter_key}, skipping")
            continue

        text = text_path.read_text(encoding="utf-8")
        sections = None
        if sections_path.exists():
            with open(sections_path, "r", encoding="utf-8") as f:
                sections = json.load(f)

        logger.info(f"Running chunking experiment for {chapter_key}")
        experiment = chunker.run_chunking_experiment(text, chapter_key, sections)

        # Save each configuration's chunks separately for retrieval
        for cfg in experiment["configurations"]:
            label = f"{cfg['strategy']}_{cfg['target_chunk_size']}"
            chunks = cfg["chunks"]
            chunker.save_chunks(chunks, chapter_key, label)
            logger.info(
                f"  {label}: {cfg['num_chunks']} chunks, "
                f"avg={cfg['avg_tokens_per_chunk']:.0f} tokens"
            )

    logger.info("✓ Stage 2 complete")
    return True


def stage_3_embed_and_retrieve():
    """
    Stage 3: Embedding, Retrieval, and Answer Generation
    
    For the best chunking configuration:
    1. Generate dense (SBERT) and sparse (TF-IDF) embeddings
    2. Build hybrid retrieval index
    3. Test retrieval with sample queries
    4. Generate grounded answers using Gemini / T5
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: Embedding, Retrieval & Generation")
    logger.info("=" * 60)

    from src.retrieval.embedder import ChunkEmbedder
    from src.retrieval.retriever import HybridRetriever
    from src.generation.answer_generator import AnswerGenerator
    from src.generation.grounding import GroundingChecker

    embedder = ChunkEmbedder()
    retriever = HybridRetriever(embedder=embedder)
    generator = AnswerGenerator()
    grounding_checker = GroundingChecker()

    # Use sentence_based @ 256 as default configuration
    # (good balance of semantic coherence and chunk size)
    default_config = "sentence_based_256"

    for chapter_key in TARGET_CHAPTERS:
        chunk_file = CHUNKING_OUTPUT_DIR / f"{chapter_key}_{default_config}_chunks.json"

        if not chunk_file.exists():
            # Try fixed_token_256 as fallback
            default_config = "fixed_token_256"
            chunk_file = CHUNKING_OUTPUT_DIR / f"{chapter_key}_{default_config}_chunks.json"
            if not chunk_file.exists():
                logger.warning(f"No chunks found for {chapter_key}, skipping")
                continue

        logger.info(f"Loading chunks: {chunk_file.name}")
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            logger.warning(f"Empty chunk file for {chapter_key}, skipping")
            continue

        # Build index
        retriever.build_index(chunks)
        retriever.save_index(chapter_key, default_config)

        # Test with sample queries
        test_queries = [
            "What is uniform motion?",
            "What is the difference between distance and displacement?",
            "How do you calculate average velocity?",
            "When is the total distance travelled equal to the magnitude of displacement?",
            "What is string theory and M-theory?", # Unknown question
        ]

        logger.info(f"\nSample retrievals for {chapter_key}:")
        for query in test_queries:
            context, results = retriever.retrieve_with_context(query, top_k=3)
            top_score = results[0]["score"] if results else 0
            logger.info(f"  Q: '{query[:50]}...' → top_score={top_score:.3f}")

            # Generate answer
            try:
                gen_result = generator.generate_answer(
                    question=query, context=context, model_type="gemini"
                )
                if gen_result["status"] == "success":
                    answer = gen_result["answer"]
                    # Check grounding
                    grounding = grounding_checker.check_grounding(answer, context)
                    logger.info(
                        f"    A: {answer[:80]}... "
                        f"(grounded={grounding['grounded']}, score={grounding['score']:.2f})"
                    )
                else:
                    logger.warning(f"    Generation failed: {gen_result.get('error', 'unknown')}")
            except Exception as e:
                logger.warning(f"    Generation error: {e}")

    # Save generation log
    log_path = RETRIEVAL_OUTPUT_DIR / "generation_log.json"
    generator.save_generation_log(log_path)

    logger.info("✓ Stage 3 complete")
    return True


def stage_4_evaluate():
    """
    Stage 4: Full System Evaluation
    
    Runs the complete evaluation pipeline on the eval set:
    1. Builds/loads evaluation question set (20+ questions, 5 types)
    2. For each question: retrieve → generate → check grounding → score
    3. Computes aggregate metrics by question type
    4. Generates detailed evaluation report
    """
    logger.info("=" * 60)
    logger.info("STAGE 4: System Evaluation")
    logger.info("=" * 60)

    from src.retrieval.embedder import ChunkEmbedder
    from src.retrieval.retriever import HybridRetriever
    from src.generation.answer_generator import AnswerGenerator
    from src.evaluation.eval_set_builder import EvalSetBuilder
    from src.evaluation.evaluator import PariShikshaEvaluator

    # Build eval set
    eval_builder = EvalSetBuilder()
    eval_set = eval_builder.build_default_eval_set()
    eval_builder.save_eval_set()
    summary = eval_builder.get_summary()
    logger.info(f"Eval set: {summary['total']} questions")
    for qtype, count in summary["by_type"].items():
        logger.info(f"  {qtype}: {count}")

    # Set up retriever (if index exists)
    default_config = "sentence_based_256"
    retriever = HybridRetriever()
    chapter_key = list(TARGET_CHAPTERS.keys())[0]  # Use first chapter

    try:
        retriever.load_index(chapter_key, default_config)
        logger.info(f"Loaded retrieval index: {chapter_key}/{default_config}")
    except Exception:
        try:
            default_config = "fixed_token_256"
            retriever.load_index(chapter_key, default_config)
            logger.info(f"Loaded fallback index: {chapter_key}/{default_config}")
        except Exception as e:
            logger.warning(f"No retrieval index found: {e}")
            logger.info("Running evaluation without retrieval (generation-only)")
            retriever = None

    # Run evaluation
    generator = AnswerGenerator()
    evaluator = PariShikshaEvaluator(
        retriever=retriever,
        generator=generator,
        eval_builder=eval_builder,
    )

    # Evaluate with Gemini
    logger.info("\nRunning evaluation with Gemini (decoder-only)...")
    report_gemini = evaluator.run_evaluation(
        eval_set=eval_set,
        model_type="gemini",
        retrieval_mode="hybrid" if retriever else "dense",
    )
    evaluator.save_report(report_gemini, "evaluation_report_gemini.json")
    evaluator.print_summary(report_gemini)

    # Evaluate with T5 (if available)
    try:
        logger.info("\nRunning evaluation with T5 (encoder-decoder)...")
        report_t5 = evaluator.run_evaluation(
            eval_set=eval_set,
            model_type="t5",
            retrieval_mode="hybrid" if retriever else "dense",
        )
        evaluator.save_report(report_t5, "evaluation_report_t5.json")
        evaluator.print_summary(report_t5)
    except Exception as e:
        logger.warning(f"T5 evaluation skipped: {e}")

    logger.info("✓ Stage 4 complete")
    return True


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="PariShiksha — Retrieval-Ready Study Assistant Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage all          Run complete pipeline
  python main.py --stage extract      Stage 1: Extraction + Tokenizer analysis
  python main.py --stage chunk        Stage 2: Chunking experiment
  python main.py --stage retrieve     Stage 3: Embedding + Retrieval + Generation
  python main.py --stage evaluate     Stage 4: Full evaluation
        """,
    )
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "chunk", "retrieve", "evaluate"],
        default="all",
        help="Which stage(s) to run",
    )

    args = parser.parse_args()

    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║         PariShiksha — Study Assistant Pipeline          ║")
    logger.info("║         Week 9 Mini-Project · PG Diploma AI-ML         ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")

    start_time = time.time()

    stages = {
        "extract": stage_1_extract_and_clean,
        "chunk": stage_2_chunking,
        "retrieve": stage_3_embed_and_retrieve,
        "evaluate": stage_4_evaluate,
    }

    if args.stage == "all":
        for name, func in stages.items():
            logger.info(f"\n{'='*60}")
            success = func()
            if not success:
                logger.error(f"Stage '{name}' failed. Stopping pipeline.")
                sys.exit(1)
    else:
        func = stages[args.stage]
        success = func()
        if not success:
            sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"\n🏁 Pipeline complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
