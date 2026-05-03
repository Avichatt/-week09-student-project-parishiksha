# =============================================================================
# PariShiksha Wk10 — Stretch Pipeline Orchestrator
# =============================================================================
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_stretch_pipeline():
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║     PariShiksha v2.0 — STRETCH TRACK Pipeline           ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    
    # Stage 1: Comparison
    try:
        from stretch_s1 import run_comparison
        run_comparison()
    except Exception as e:
        logger.error(f"Stage 1 failed: {e}")

    # Stage 2: Benchmark
    try:
        from stretch_s2 import BenchmarkEngine
        engine = BenchmarkEngine()
        engine.run_benchmark()
    except Exception as e:
        logger.error(f"Stage 2 failed: {e}")

    # Stage 3: Hybrid
    try:
        from stretch_s3 import run_stretch_stage3
        run_stretch_stage3()
    except Exception as e:
        logger.error(f"Stage 3 failed: {e}")

    # Stage 4: RAGAS + Rerank
    try:
        from stretch_s4 import run_stretch_stage4
        run_stretch_stage4()
    except Exception as e:
        logger.error(f"Stage 4 failed: {e}")

    logger.info("🏁 Stretch Pipeline complete. See docs/ for artifacts.")

if __name__ == "__main__":
    run_stretch_pipeline()
