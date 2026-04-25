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
        self.chroma_client = None
        self.collection = None
        self.bm25 = None
        self.cross_encoder = None

    # -------------------------------------------------------------------------
    # Index Management
    # -------------------------------------------------------------------------

    def _init_bm25(self) -> None:
        """Initialize BM25 index for keyword search."""
        if not self.chunks:
            return
            
        from rank_bm25 import BM25Okapi
        # Simple tokenization by splitting on whitespace/lowercase
        corpus = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(corpus)} documents")

    def _init_cross_encoder(self) -> None:
        """Initialize Cross-Encoder for high-precision re-ranking."""
        if self.cross_encoder is not None:
            return
            
        from sentence_transformers import CrossEncoder
        model_name = self.config.get("cross_encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        logger.info(f"Loading Cross-Encoder re-ranker: {model_name}")
        self.cross_encoder = CrossEncoder(model_name)

    def _init_chroma(self, chapter_key: str) -> None:
        """Initialize ChromaDB collection for dense retrieval."""
        import chromadb
        from chromadb.config import Settings
        
        persist_dir = self.config.get("retrieval_output_dir", RETRIEVAL_OUTPUT_DIR) / "chroma_db"
        self.chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        
        # We use one collection per chapter or a shared one with metadata filters
        collection_name = f"parishiksha_{chapter_key}"
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )
        logger.info(f"ChromaDB collection '{collection_name}' initialized at {persist_dir}")

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Build retrieval index from chunks and ingest into ChromaDB.
        """
        logger.info(f"Building retrieval index from {len(chunks)} chunks")
        self.chunks = chunks
        
        # 1. Dense Embeddings
        self.embedder.embed_dense(chunks)
        
        # 2. Sparse Index (Brain)
        self._init_bm25()
        
        # Note: Ingestion happens in save_index or directly here
        self.is_loaded = True
        logger.info("Index built in memory")

    def save_index(self, chapter_key: str, config_label: str) -> None:
        """Save index to ChromaDB (Memory) and local files (Brain)."""
        self.embedder.save_embeddings(chapter_key, config_label)
        
        # Initialize Chroma for this specific chapter
        self._init_chroma(f"{chapter_key}_{config_label}")
        
        # Upsert data into ChromaDB
        if self.embedder.dense_embeddings is not None:
            # Prepare data
            documents = [c["text"] for c in self.chunks]
            # Convert np.ndarray to list of lists for Chroma
            embeddings = self.embedder.dense_embeddings.tolist()
            # Ensure metadatas are clean dicts AND contain chapter info (Gap 4)
            metadatas = []
            for c in self.chunks:
                meta = c.get("metadata", {}).copy()
                meta["chapter"] = chapter_key # Inject identifier
                metadatas.append(meta)
            
            # Chroma IDs must be strings
            ids = [f"c_{i}" for i in range(len(self.chunks))]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Ingested {len(ids)} chunks into ChromaDB collection")

    def load_index(self, chapter_key: str, config_label: str) -> None:
        """Load a previously saved index from ChromaDB."""
        self.embedder.load_embeddings(chapter_key, config_label)
        self.chunks = self.embedder.chunks
        
        # Connect to Chroma collection
        self._init_chroma(f"{chapter_key}_{config_label}")
        
        self._init_bm25()
        self.is_loaded = True
        logger.info(f"Industrial Index Loaded: {chapter_key}/{config_label}")

    # -------------------------------------------------------------------------
    # Retrieval
    # -------------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        chapter_filter: Optional[str] = None,
        do_rerank: bool = True,
    ) -> List[Dict]:
        """
        Industrial-grade retrieval: Normalize -> Filter -> Hybrid Search -> Deduplicate -> Re-rank.
        """
        if not self.is_loaded:
            raise RuntimeError("No index loaded.")

        top_k = top_k or self.top_k
        # Stage 1: Broad candidate retrieval (Chroma + Metadata Filter)
        candidate_k = 50 
        
        # 1. Query Normalization (Gap 7)
        normalized_query = query.lower().strip()
        
        # 2. Dense Candidate Fetch (Gap 1, 2)
        query_dense = self.embedder.embed_query_dense(normalized_query).tolist()
        
        where_clause = {"chapter": chapter_filter} if chapter_filter else None
        
        dense_results = self.collection.query(
            query_embeddings=[query_dense],
            n_results=candidate_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Map Chroma results to our internal format
        candidates = []
        for i, (doc, meta, dist, id_str) in enumerate(zip(
            dense_results["documents"][0],
            dense_results["metadatas"][0],
            dense_results["distances"][0],
            dense_results["ids"][0]
        )):
            candidates.append({
                "chunk_id": id_str,
                "text": doc,
                "dense_score": 1.0 - dist,
                "metadata": meta,
                "temp_idx": int(id_str.split("_")[1])
            })

        # 3. Efficient Sparse Re-scoring (Gap 3)
        # We only score the 50 candidates fetched by dense search
        if self.bm25 is not None and mode in ("hybrid", "sparse"):
            tokenized_query = normalized_query.split()
            # Calculate all scores once for the entire corpus
            full_bm25_scores = self.bm25.get_scores(tokenized_query)
            
            for cand in candidates:
                actual_idx = cand["temp_idx"]
                # Safeguard against index out of bounds if chunks changed
                if actual_idx < len(full_bm25_scores):
                    cand["sparse_score"] = float(full_bm25_scores[actual_idx])
                else:
                    cand["sparse_score"] = 0.0
        else:
            for cand in candidates: cand["sparse_score"] = 0.0

        # 4. Hybrid Combination & Dedup (Gap 5, 8)
        seen_texts = set()
        final_candidates = []
        
        # Normalize sparse scores among candidates for stable hybrid weighting
        max_sparse = max([c["sparse_score"] for c in candidates]) if candidates else 0
        
        for cand in candidates:
            # Deduplication
            text_hash = cand["text"].strip().lower()
            if text_hash in seen_texts: continue
            seen_texts.add(text_hash)
            
            # Combine scores
            norm_sparse = cand["sparse_score"] / max_sparse if max_sparse > 0 else 0
            cand["combined_score"] = self.alpha * cand["dense_score"] + (1 - self.alpha) * norm_sparse
            # Ensure a 'score' key exists even before re-ranking
            cand["score"] = cand["combined_score"]
            final_candidates.append(cand)

        # Sort by hybrid score
        final_candidates = sorted(final_candidates, key=lambda x: x["combined_score"], reverse=True)
        
        # 5. Semantic Re-ranking (Stage 3)
        initial_top = final_candidates[:top_k * 3]
        if do_rerank and self.cross_encoder and len(initial_top) > 1:
            try:
                pairs = [[normalized_query, c["text"]] for c in initial_top]
                rerank_scores = self.cross_encoder.predict(pairs)
                for i, score in enumerate(rerank_scores):
                    initial_top[i]["score"] = float(score)
                initial_top = sorted(initial_top, key=lambda x: x["score"], reverse=True)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")

        return initial_top[:top_k]

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        chapter_filter: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve chunks and format them as a context string for generation.
        """
        results = self.retrieve(query, top_k=top_k, mode=mode, chapter_filter=chapter_filter)

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
