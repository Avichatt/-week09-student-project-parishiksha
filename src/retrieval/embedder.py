# =============================================================================
# PariShiksha — Chunk Embedding Module
# =============================================================================
# Converts text chunks into dense vector embeddings using Sentence-BERT.
# Also provides TF-IDF sparse representations as a baseline comparison.
#
# This is the bridge between chunking (Stage 2) and retrieval (Stage 3).
# The embedding model's tokenizer is different from the chunking tokenizer,
# so chunks that were "256 BERT tokens" might be a different count under
# SBERT's tokenizer. This is a real-world gotcha that many tutorials skip.
# =============================================================================

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import EMBEDDING_CONFIG, CHUNKING_OUTPUT_DIR, RETRIEVAL_OUTPUT_DIR


class ChunkEmbedder:
    """
    Embeds text chunks using dense (SBERT) and sparse (TF-IDF) representations.
    
    Dense embeddings (SBERT):
    - Capture semantic meaning
    - Handle paraphrased queries well
    - May miss exact keyword matches
    
    Sparse embeddings (TF-IDF):
    - Excellent for exact term matching
    - Fast and interpretable
    - Cannot handle paraphrasing
    
    Using both gives us a hybrid search capability (Stage 3).
    
    Usage:
        embedder = ChunkEmbedder()
        chunks = json.load(open("chapter_5_fixed_token_256_chunks.json"))
        
        # Dense embeddings
        dense = embedder.embed_dense(chunks)
        
        # Sparse embeddings
        sparse = embedder.embed_sparse(chunks)
        
        # Save for retrieval
        embedder.save_embeddings("chapter_5", "fixed_token_256")
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        output_dir: Path = RETRIEVAL_OUTPUT_DIR,
    ):
        self.config = config or EMBEDDING_CONFIG
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.dense_model = None
        self.tfidf_vectorizer = None
        self.dense_embeddings = None
        self.sparse_embeddings = None
        self.chunks = None

    # -------------------------------------------------------------------------
    # Dense Embeddings (SBERT)
    # -------------------------------------------------------------------------

    def embed_dense(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate dense embeddings using Sentence-BERT.
        
        Parameters
        ----------
        chunks : list of dict
            Each chunk must have a "text" field.
            
        Returns
        -------
        np.ndarray
            Shape (n_chunks, embedding_dim). For all-MiniLM-L6-v2, dim=384.
        """
        from sentence_transformers import SentenceTransformer

        if self.dense_model is None:
            model_name = self.config.get("dense_model", "all-MiniLM-L6-v2")
            logger.info(f"Loading SBERT model: {model_name}")
            self.dense_model = SentenceTransformer(model_name)

        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        logger.info(f"Embedding {len(texts)} chunks with SBERT...")
        self.dense_embeddings = self.dense_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
        )

        logger.info(
            f"Dense embeddings shape: {self.dense_embeddings.shape} "
            f"(dim={self.dense_embeddings.shape[1]})"
        )
        return self.dense_embeddings

    def embed_query_dense(self, query: str) -> np.ndarray:
        """Embed a single query for dense retrieval."""
        if self.dense_model is None:
            raise RuntimeError("Call embed_dense() first to load the model")
        return self.dense_model.encode(
            [query], normalize_embeddings=True
        )[0]

    # -------------------------------------------------------------------------
    # Sparse Embeddings (TF-IDF)
    # -------------------------------------------------------------------------

    def embed_sparse(self, chunks: List[Dict]) -> object:
        """
        Generate TF-IDF sparse representations.
        
        Parameters
        ----------
        chunks : list of dict
            Each chunk must have a "text" field.
            
        Returns
        -------
        scipy.sparse matrix
            Shape (n_chunks, n_features)
        """
        self.chunks = chunks
        texts = [chunk["text"] for chunk in chunks]

        max_features = self.config.get("tfidf_max_features", 5000)
        ngram_range = tuple(self.config.get("tfidf_ngram_range", (1, 2)))

        logger.info(f"Building TF-IDF matrix (max_features={max_features}, ngrams={ngram_range})")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,  # Apply sublinear tf scaling (1 + log(tf))
        )

        self.sparse_embeddings = self.tfidf_vectorizer.fit_transform(texts)

        logger.info(
            f"TF-IDF matrix shape: {self.sparse_embeddings.shape} "
            f"(vocab size={len(self.tfidf_vectorizer.vocabulary_)})"
        )
        return self.sparse_embeddings

    def embed_query_sparse(self, query: str) -> object:
        """Transform a query into TF-IDF representation."""
        if self.tfidf_vectorizer is None:
            raise RuntimeError("Call embed_sparse() first to fit the vectorizer")
        return self.tfidf_vectorizer.transform([query])

    # -------------------------------------------------------------------------
    # I/O
    # -------------------------------------------------------------------------

    def save_embeddings(self, chapter_key: str, config_label: str) -> Dict[str, Path]:
        """
        Save embeddings and metadata for retrieval.
        
        Saves:
        - Dense embeddings as .npy
        - Sparse embeddings as .pkl (scipy sparse matrix)
        - TF-IDF vectorizer as .pkl
        - Chunk metadata as .json
        """
        prefix = f"{chapter_key}_{config_label}"
        paths = {}

        if self.dense_embeddings is not None:
            path = self.output_dir / f"{prefix}_dense.npy"
            np.save(str(path), self.dense_embeddings)
            paths["dense"] = path
            logger.info(f"Saved dense embeddings: {path}")

        if self.sparse_embeddings is not None:
            path = self.output_dir / f"{prefix}_sparse.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.sparse_embeddings, f)
            paths["sparse"] = path
            logger.info(f"Saved sparse embeddings: {path}")

        if self.tfidf_vectorizer is not None:
            path = self.output_dir / f"{prefix}_tfidf_vectorizer.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)
            paths["vectorizer"] = path

        if self.chunks is not None:
            path = self.output_dir / f"{prefix}_chunk_metadata.json"
            # Save chunk texts and metadata (without embeddings)
            chunk_data = [
                {"chunk_id": c.get("chunk_id", i), "text": c["text"],
                 "token_count": c.get("token_count", 0),
                 "metadata": c.get("metadata", {})}
                for i, c in enumerate(self.chunks)
            ]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            paths["metadata"] = path

        return paths

    def load_embeddings(self, chapter_key: str, config_label: str) -> None:
        """Load previously saved embeddings."""
        prefix = f"{chapter_key}_{config_label}"

        dense_path = self.output_dir / f"{prefix}_dense.npy"
        if dense_path.exists():
            self.dense_embeddings = np.load(str(dense_path))
            logger.info(f"Loaded dense embeddings: {self.dense_embeddings.shape}")

        sparse_path = self.output_dir / f"{prefix}_sparse.pkl"
        if sparse_path.exists():
            with open(sparse_path, "rb") as f:
                self.sparse_embeddings = pickle.load(f)
            logger.info(f"Loaded sparse embeddings: {self.sparse_embeddings.shape}")

        vectorizer_path = self.output_dir / f"{prefix}_tfidf_vectorizer.pkl"
        if vectorizer_path.exists():
            with open(vectorizer_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

        metadata_path = self.output_dir / f"{prefix}_chunk_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunk metadata entries")


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    embedder = ChunkEmbedder()

    # Try to load chunks from chunking output
    for chunk_file in CHUNKING_OUTPUT_DIR.glob("*_chunks.json"):
        print(f"\nProcessing: {chunk_file.name}")
        with open(chunk_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        if not chunks:
            print("  No chunks found, skipping")
            continue

        # Generate embeddings
        dense = embedder.embed_dense(chunks)
        sparse = embedder.embed_sparse(chunks)

        # Derive labels from filename
        parts = chunk_file.stem.split("_")
        chapter_key = "_".join(parts[:2])
        config_label = "_".join(parts[2:]).replace("_chunks", "")

        # Save
        paths = embedder.save_embeddings(chapter_key, config_label)
        print(f"  Saved to: {list(paths.values())}")
