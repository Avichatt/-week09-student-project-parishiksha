# =============================================================================
# PariShiksha Wk10 — Stage 2: OpenAI Embedding + ChromaDB Persistence
# =============================================================================
# Rubric requirements:
#   - Embed wk10_chunks.json with OpenAI text-embedding-3-small
#   - Persist to Chroma (PersistentClient, path ./chroma_wk10)
#   - Use cosine similarity
#   - Don't re-embed every kernel restart
#   - Build retrieve(query, k=5) returning chunks with similarity scores
# =============================================================================

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chromadb
from loguru import logger
from dotenv import load_dotenv

load_dotenv()


class Wk10Embedder:
    """
    Embeds chunks using OpenAI text-embedding-3-small and persists to ChromaDB.
    
    Key design decisions:
    - Uses OpenAI's text-embedding-3-small (1536-dim) per rubric
    - ChromaDB PersistentClient at ./chroma_wk10 for persistence
    - Cosine similarity for retrieval
    - Skips re-embedding if collection already populated
    """

    COLLECTION_NAME = "parishiksha_wk10"

    def __init__(self, chroma_path: str = "./chroma_wk10"):
        self.chroma_path = chroma_path
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = None
        self.chunks = []
        self._openai_client = None

    def _get_openai_client(self):
        """Lazy-init OpenAI client."""
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY not set. Add it to .env file."
                )
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts using OpenAI text-embedding-3-small."""
        client = self._get_openai_client()
        
        # OpenAI allows max 2048 texts per batch, chunk if needed
        all_embeddings = []
        batch_size = 100
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"  Embedded batch {i//batch_size + 1}: {len(batch)} texts")
        
        return all_embeddings

    def load_and_embed(self, chunks_path: str = "wk10_chunks.json") -> None:
        """
        Load chunks and embed into ChromaDB.
        Skips if collection already has the right number of documents.
        """
        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        
        logger.info(f"Loaded {len(self.chunks)} chunks from {chunks_path}")

        # Get or create collection with cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        existing_count = self.collection.count()
        logger.info(f"ChromaDB collection '{self.COLLECTION_NAME}' has {existing_count} docs")

        if existing_count == len(self.chunks):
            logger.info("Collection already populated. Skipping re-embedding.")
            return

        # Need to embed — clear and re-populate
        if existing_count > 0:
            logger.info("Collection size mismatch. Clearing and re-embedding.")
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Prepare data
        texts = [c["text"] for c in self.chunks]
        ids = [c["chunk_id"] for c in self.chunks]
        
        # Clean metadata for ChromaDB (only simple types)
        metadatas = []
        for c in self.chunks:
            meta = {
                "source": c["metadata"].get("source", ""),
                "section": c["metadata"].get("section", ""),
                "content_type": c["metadata"].get("content_type", "prose"),
                "page": c["metadata"].get("page", 0),
                "chapter": c["metadata"].get("chapter", "chapter_4"),
                "chunk_index": c["metadata"].get("chunk_index", 0),
                "token_count": c.get("token_count", 0),
            }
            metadatas.append(meta)

        # Embed with OpenAI
        logger.info("Embedding with OpenAI text-embedding-3-small...")
        embeddings = self._embed_texts(texts)
        logger.info(f"Generated {len(embeddings)} embeddings (dim={len(embeddings[0])})")

        # Upsert into ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(f"Ingested {len(ids)} chunks into ChromaDB at {self.chroma_path}")

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k chunks for a query with similarity scores.
        
        Returns list of dicts with: chunk_id, text, score, metadata
        """
        if self.collection is None:
            self.collection = self.client.get_collection(self.COLLECTION_NAME)

        # Embed query
        query_embedding = self._embed_texts([query])[0]

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        retrieved = []
        for i in range(len(results["ids"][0])):
            # ChromaDB returns distances; for cosine, score = 1 - distance
            distance = results["distances"][0][i]
            score = 1.0 - distance

            retrieved.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "score": round(score, 4),
                "metadata": results["metadatas"][0][i],
            })

        return retrieved

    def retrieve_with_context(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        """Retrieve and format as context string for generation."""
        results = self.retrieve(query, k=k)
        
        context_parts = []
        for i, r in enumerate(results, 1):
            context_parts.append(
                f"[Source: {r['chunk_id']}] (section: {r['metadata'].get('section', 'N/A')}, "
                f"score: {r['score']:.3f})\n{r['text']}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        return context, results


# =============================================================================
# CLI Entry Point
# =============================================================================
if __name__ == "__main__":
    embedder = Wk10Embedder()
    
    chunks_path = Path("wk10_chunks.json")
    if not chunks_path.exists():
        print("Run wk10_chunker.py first to generate wk10_chunks.json")
        exit(1)
    
    embedder.load_and_embed(str(chunks_path))
    
    # Test retrieval
    test_queries = [
        "What is displacement?",
        "Define uniform motion.",
        "What are the equations of motion?",
        "What is uniform circular motion?",
        "How to calculate average acceleration?",
    ]
    
    print(f"\n{'='*60}")
    print("RETRIEVAL TEST (k=5)")
    print(f"{'='*60}")
    
    for query in test_queries:
        results = embedder.retrieve(query, k=5)
        top = results[0] if results else None
        print(f"\nQ: {query}")
        if top:
            print(f"  Top-1: [{top['chunk_id']}] score={top['score']:.4f}")
            print(f"  Section: {top['metadata'].get('section', 'N/A')}")
            print(f"  Type: {top['metadata'].get('content_type', 'N/A')}")
            print(f"  Text: {top['text'][:120]}...")
