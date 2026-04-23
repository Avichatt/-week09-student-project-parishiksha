# =============================================================================
# PariShiksha — Hybrid Retrieval Module
# =============================================================================
# Retrieves the most relevant chunks for a given query using a combination
# of dense (SBERT) and sparse (TF-IDF) similarity.
#
# Why hybrid?
# -----------
# Dense retrieval handles paraphrased queries ("What makes plants green?" → chlorophyll)
# but can miss exact terms. Sparse retrieval catches exact matches ("chlorophyll function")
# but cannot handle semantic paraphrasing. Combining both gives robust retrieval
# across the range of queries students actually ask.
# =============================================================================

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import EMBEDDING_CONFIG, RETRIEVAL_OUTPUT_DIR
from src.retrieval.embedder import ChunkEmbedder


class HybridRetriever:
    """
    Retrieves relevant chunks using hybrid dense + sparse scoring.
    
    Scoring formula:
        score = alpha * dense_score + (1 - alpha) * sparse_score
    
    where alpha controls the dense/sparse trade-off.
    
    Usage:
        retriever = HybridRetriever()
        retriever.load_index("chapter_5", "fixed_token_256")
        results = retriever.retrieve("What is the function of mitochondria?", top_k=5)
    """

    def __init__(
        self,
        embedder: Optional[ChunkEmbedder] = None,
        config: Optional[Dict] = None,
        alpha: float = 0.7,  # weight for dense score
    ):
        self.embedder = embedder or ChunkEmbedder()
        self.config = config or EMBEDDING_CONFIG
        self.alpha = alpha
        self.top_k = self.config.get("top_k", 5)

        # State
        self.chunks = None
        self.is_loaded = False

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build retrieval index from chunks.
        
        Generates both dense and sparse embeddings.
        """
        logger.info(f"Building retrieval index from {len(chunks)} chunks")
        self.embedder.embed_dense(chunks)
        self.embedder.embed_sparse(chunks)
        self.chunks = chunks
        self.is_loaded = True
        logger.info("Index built successfully")

    def save_index(self, chapter_key: str, config_label: str) -> None:
        """Save index for later use."""
        self.embedder.save_embeddings(chapter_key, config_label)

    def load_index(self, chapter_key: str, config_label: str) -> None:
        """Load a previously saved index."""
        self.embedder.load_embeddings(chapter_key, config_label)
        self.chunks = self.embedder.chunks
        self.is_loaded = True
        logger.info(f"Index loaded: {chapter_key}/{config_label}")

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
    ) -> List[Dict]:
        """
        Retrieve top-k relevant chunks for a query.
        
        Parameters
        ----------
        query : str
            The student's question
        top_k : int, optional
            Number of chunks to retrieve. Defaults to config value.
        mode : str
            Retrieval mode: "hybrid", "dense", or "sparse"
            
        Returns
        -------
        list of dict
            Ranked results with scores:
            [{"chunk_id": int, "text": str, "score": float, "dense_score": float,
              "sparse_score": float, "metadata": dict}, ...]
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded. Call build_index() or load_index() first.")

        top_k = top_k or self.top_k

        # Compute dense scores
        dense_scores = np.zeros(len(self.chunks))
        if mode in ("hybrid", "dense") and self.embedder.dense_embeddings is not None:
            query_dense = self.embedder.embed_query_dense(query)
            dense_scores = cosine_similarity(
                query_dense.reshape(1, -1),
                self.embedder.dense_embeddings
            )[0]

        # Compute sparse scores
        sparse_scores = np.zeros(len(self.chunks))
        if mode in ("hybrid", "sparse") and self.embedder.sparse_embeddings is not None:
            query_sparse = self.embedder.embed_query_sparse(query)
            sparse_scores = cosine_similarity(
                query_sparse,
                self.embedder.sparse_embeddings
            )[0]

        # Combine scores
        if mode == "hybrid":
            combined_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
        elif mode == "dense":
            combined_scores = dense_scores
        else:
            combined_scores = sparse_scores

        # Rank and return top-k
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.get("chunk_id", idx),
                "text": chunk["text"],
                "score": float(combined_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "token_count": chunk.get("token_count", 0),
                "metadata": chunk.get("metadata", {}),
            })

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve chunks and format them as a context string for generation.
        
        Returns
        -------
        tuple of (context_str, results)
            context_str: formatted context for the LLM prompt
            results: ranked retrieval results
        """
        results = self.retrieve(query, top_k=top_k, mode=mode)

        # Format context for generation prompt
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Context {i}] (relevance: {result['score']:.3f})\n{result['text']}"
            )

        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, results

    # -------------------------------------------------------------------------
    # Evaluation Helpers
    # -------------------------------------------------------------------------

    def evaluate_retrieval(
        self,
        queries_with_expected: List[Dict],
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> Dict:
        """
        Evaluate retrieval quality on a set of queries with expected content.
        
        Parameters
        ----------
        queries_with_expected : list of dict
            Each entry: {"query": str, "expected_keywords": [str], 
                         "expected_section": str (optional)}
            
        Returns
        -------
        dict
            Evaluation metrics: precision@k, recall of keywords, MRR
        """
        results = {
            "per_query": [],
            "aggregate": {},
        }

        all_precisions = []
        all_mrr = []
        all_keyword_recalls = []

        for item in queries_with_expected:
            query = item["query"]
            expected_keywords = item.get("expected_keywords", [])
            expected_section = item.get("expected_section", "")

            retrieved = self.retrieve(query, top_k=top_k, mode=mode)

            # Keyword recall: what fraction of expected keywords appear in retrieved chunks?
            retrieved_text = " ".join(r["text"].lower() for r in retrieved)
            keywords_found = sum(
                1 for kw in expected_keywords
                if kw.lower() in retrieved_text
            )
            keyword_recall = keywords_found / len(expected_keywords) if expected_keywords else 0

            # Section precision: does the expected section appear in any retrieved chunk?
            section_hit = False
            first_section_rank = -1
            if expected_section:
                for i, r in enumerate(retrieved):
                    chunk_section = r.get("metadata", {}).get("section_heading", "")
                    if expected_section.lower() in chunk_section.lower():
                        section_hit = True
                        if first_section_rank == -1:
                            first_section_rank = i + 1
                        break

            # MRR (Mean Reciprocal Rank) based on section match
            mrr = 1 / first_section_rank if first_section_rank > 0 else 0

            query_result = {
                "query": query,
                "keyword_recall": round(keyword_recall, 3),
                "section_hit": section_hit,
                "mrr": round(mrr, 3),
                "top_scores": [round(r["score"], 3) for r in retrieved[:3]],
            }
            results["per_query"].append(query_result)
            all_keyword_recalls.append(keyword_recall)
            all_mrr.append(mrr)

        # Aggregate
        results["aggregate"] = {
            "mean_keyword_recall": round(np.mean(all_keyword_recalls), 3) if all_keyword_recalls else 0,
            "mean_mrr": round(np.mean(all_mrr), 3) if all_mrr else 0,
            "queries_evaluated": len(queries_with_expected),
        }

        return results


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    retriever = HybridRetriever()

    # Try to load an existing index
    try:
        retriever.load_index("chapter_5", "fixed_token_256")
    except Exception as e:
        print(f"Could not load index: {e}")
        print("Run embedder first to build the index.")
        exit(1)

    # Interactive query loop
    print("\n" + "=" * 60)
    print("PariShiksha Retrieval Demo")
    print("Type a question (or 'quit' to exit)")
    print("=" * 60)

    while True:
        query = input("\n> ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break

        context, results = retriever.retrieve_with_context(query)
        print(f"\nRetrieved {len(results)} chunks:")
        for r in results:
            print(f"\n  [{r['chunk_id']}] Score: {r['score']:.3f} "
                  f"(dense: {r['dense_score']:.3f}, sparse: {r['sparse_score']:.3f})")
            print(f"  {r['text'][:200]}...")
