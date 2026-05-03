# =============================================================================
# PariShiksha Wk10 — Stretch Stage 3: Hybrid Retrieval & 20-Q Eval
# =============================================================================
# Rubric requirements:
#   - Hybrid retrieval: BM25 + Dense (Fused by RRF)
#   - Strict prompt with citations
#   - 20-question eval set (10 direct, 5 paraphrased, 5 OOS)
#   - Identify 2 failure patterns
#   - Save eval_v3_scored.csv
# =============================================================================

import json
import os
from typing import Dict, List, Tuple

from loguru import logger
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

from retrieval import Wk10Embedder
from generation import Wk10AskEngine, STRICT_PROMPT

class HybridRetriever:
    """Combines BM25 and Vector Search using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, chunks_path: str = "data/results/wk10_chunks.json"):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        self.texts = [c["text"] for c in self.chunks]
        self.ids = [c["chunk_id"] for c in self.chunks]
        
        # Initialize BM25
        tokenized_corpus = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Initialize Dense (uses Wk10Embedder logic)
        self.embedder = Wk10Embedder()
        self.embedder.collection = self.embedder.client.get_or_create_collection(
            name=Wk10Embedder.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def retrieve(self, query: str, k: int = 5, b: int = 60) -> List[Dict]:
        """Retrieve top-k chunks using RRF."""
        # 1. BM25 results
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_rank = np.argsort(bm25_scores)[::-1]
        
        # 2. Dense results
        dense_results = self.embedder.retrieve(query, k=20) # Get more candidates for fusion
        dense_rank_map = {res["chunk_id"]: i for i, res in enumerate(dense_results)}
        
        # 3. RRF Fusion
        rrf_scores = {}
        
        # BM25 contribution
        for rank, idx in enumerate(bm25_rank[:20]):
            cid = self.ids[idx]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + b)
            
        # Dense contribution
        for cid, rank in dense_rank_map.items():
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rank + b)
            
        # Sort and get top-k
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        top_ids = sorted_ids[:k]
        
        # Build results
        results = []
        chunk_map = {c["chunk_id"]: c for c in self.chunks}
        for cid in top_ids:
            chunk = chunk_map[cid]
            results.append({
                "chunk_id": cid,
                "text": chunk["text"],
                "score": round(rrf_scores[cid], 4),
                "metadata": chunk.get("metadata", {})
            })
            
        return results

# ---------------------------------------------------------------------------
# Expanded Eval Set (20 Questions)
# ---------------------------------------------------------------------------
STRETCH_EVAL_SET = [
    # --- 10 DIRECT ---
    {"id": "D1", "q": "What is displacement?", "type": "direct"},
    {"id": "D2", "q": "Definition of average speed.", "type": "direct"},
    {"id": "D3", "q": "Formula for average acceleration.", "type": "direct"},
    {"id": "D4", "q": "What are the equations of motion?", "type": "direct"},
    {"id": "D5", "q": "Define uniform circular motion.", "type": "direct"},
    {"id": "D6", "q": "What is the SI unit of velocity?", "type": "direct"},
    {"id": "D7", "q": "What is the slope of a distance-time graph?", "type": "direct"},
    {"id": "D8", "q": "What is the area under a velocity-time graph?", "type": "direct"},
    {"id": "D9", "q": "Define retardation.", "type": "direct"},
    {"id": "D10", "q": "What is the magnitude of g on Earth?", "type": "direct"},
    
    # --- 5 PARAPHRASED ---
    {"id": "P1", "q": "If I return to my starting point, what is my displacement?", "type": "paraphrased"},
    {"id": "P2", "q": "How is slowing down represented in physics?", "type": "paraphrased"},
    {"id": "P3", "q": "Can speed be constant while velocity changes?", "type": "paraphrased"},
    {"id": "P4", "q": "What does a speedometer show?", "type": "paraphrased"},
    {"id": "P5", "q": "Difference between scalar and vector in motion.", "type": "paraphrased"},
    
    # --- 5 OOS (2 plausibly answerable) ---
    {"id": "OOS1", "q": "What is the speed of light?", "type": "oos"},
    {"id": "OOS2", "q": "How do plants make food?", "type": "oos"},
    {"id": "OOS3", "q": "What is the gravity on Mars?", "type": "oos"}, # Plausibly answerable
    {"id": "OOS4", "q": "Calculate g on the Moon.", "type": "oos"}, # Plausibly answerable
    {"id": "OOS5", "q": "Who wrote the theory of relativity?", "type": "oos"},
]

import numpy as np

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_stretch_stage3():
    logger.info("=== Stretch Stage 3: Hybrid Retrieval & 20-Q Eval ===")
    retriever = HybridRetriever()
    engine = Wk10AskEngine(prompt_mode="strict")
    # Override retriever with hybrid
    engine.embedder.retrieve = retriever.retrieve 

    results = []
    
    for item in STRETCH_EVAL_SET:
        logger.info(f"[{item['id']}] Asking: {item['q']}")
        try:
            # We use a try-except because we know Gemini quota might be hit
            res = engine.ask(item['q'])
            results.append({
                "id": item["id"],
                "question": item["q"],
                "type": item["type"],
                "answer": res["answer"],
                "top_chunk": res["chunk_ids"][0] if res["chunk_ids"] else "N/A"
            })
        except Exception as e:
            logger.error(f"  Failed: {e}")
            results.append({
                "id": item["id"],
                "question": item["q"],
                "type": item["type"],
                "answer": f"ERROR: {e}",
                "top_chunk": "N/A"
            })

    # Save CSV
    import csv
    with open("eval_v3_scored.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "type", "answer", "top_chunk"])
        writer.writeheader()
        writer.writerows(results)
    
    logger.info("Saved eval_v3_scored.csv")

if __name__ == "__main__":
    run_stretch_stage3()
