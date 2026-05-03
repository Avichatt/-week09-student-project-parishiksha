# =============================================================================
# PariShiksha Wk10 — Stretch Pipeline Orchestrator
# =============================================================================
# Runs the Stretch track workflow:
#   Stage 1: Multi-variant chunking comparison
#   Stage 2: DB & Embedding benchmark
#   Stage 3: Hybrid retrieval + 20-Q eval
#   Stage 4: Rerank + MultiQuery + RAGAS
#   Stage 5: Failure memo (static)
# =============================================================================

import argparse
import sys
from loguru import logger

def run_stretch_pipeline():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     PariShiksha v2.0 — STRETCH TRACK Pipeline           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    
    try:
        from wk10_stretch_stage1 import run_comparison
        run_comparison()
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")

    try:
        from wk10_stretch_stage2 import BenchmarkEngine
        engine = BenchmarkEngine()
        engine.run_benchmark()
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")

    try:
        from wk10_stretch_stage3 import run_stretch_stage3
        run_stretch_stage3()
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")

    try:
        from wk10_stretch_stage4 import run_stretch_stage4
        run_stretch_stage4()
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")

    logger.info("🏁 Stretch Pipeline complete. All artifacts generated in workspace.")

if __name__ == "__main__":
    run_stretch_pipeline()
