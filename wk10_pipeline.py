# =============================================================================
# PariShiksha Wk10 — Main Pipeline Orchestrator (v2.0)
# =============================================================================
import argparse
import sys
import time
from pathlib import Path
from loguru import logger

# Add src to path for industry-standard module resolution
sys.path.append(str(Path(__file__).parent / "src"))

from chunking import Wk10Chunker
from retrieval import Wk10Embedder
from generation import GeminiGenerator # Note: Renamed in move? No, I named it generation.py
from evaluation import Wk10Evaluator

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<7} | {message}")
logger.add("data/results/wk10_pipeline.log", level="DEBUG", rotation="10 MB")

def stage_1_chunk():
    logger.info("STAGE 1: Content-Type-Aware Chunking")
    chunker = Wk10Chunker(max_tokens=250)
    sections_path = "data/processed/chapter_4_sections.json"
    chunks = chunker.chunk_sections_from_file(sections_path)
    
    output_path = "data/results/wk10_chunks.json"
    chunker.save_chunks(chunks, output_path)
    
    evaluator = Wk10Evaluator()
    evaluator.generate_chunking_diff(output_path, "docs/chunking_diff.md")
    return True

def stage_2_embed():
    logger.info("STAGE 2: Google Embedding + ChromaDB")
    embedder = Wk10Embedder(persist_directory="./storage/chroma_wk10")
    chunks_path = "data/results/wk10_chunks.json"
    embedder.load_and_embed(chunks_path)
    
    queries = ["What is displacement?", "Define average speed.", "How is acceleration calculated?"]
    log_path = "data/results/retrieval_log.json"
    embedder.generate_retrieval_log(queries, log_path)
    
    evaluator = Wk10Evaluator()
    evaluator.generate_miss_report(log_path, "docs/retrieval_misses.md")
    return True

def stage_3_generate():
    logger.info("STAGE 3: Grounded Generation (Gemini 2.0)")
    # Logic moved to src/generation.py
    return True

def stage_4_5_evaluate():
    logger.info("STAGE 4+5: Evaluation + Targeted Fix")
    evaluator = Wk10Evaluator()
    evaluator.run_full_evaluation()
    return True

def main():
    parser = argparse.ArgumentParser(description="PariShiksha v2.0 — Wk10 Pipeline")
    parser.add_argument("--stage", choices=["all", "chunk", "embed", "generate", "evaluate"], default="all")
    args = parser.parse_args()
    
    stages = {
        "chunk": stage_1_chunk,
        "embed": stage_2_embed,
        "generate": stage_3_generate,
        "evaluate": stage_4_5_evaluate,
    }
    
    if args.stage == "all":
        for name, func in stages.items():
            if not func(): break
    else:
        stages[args.stage]()

if __name__ == "__main__":
    main()
