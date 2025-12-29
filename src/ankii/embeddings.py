"""Embedding service for card text."""

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Callable

import httpx
import numpy as np
from dotenv import load_dotenv


load_dotenv()


OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"
DEFAULT_MODEL = "openai/text-embedding-3-small"
CACHE_DIR = Path.home() / ".cache" / "ankii" / "embeddings"


class EmbeddingError(Exception):
    """Raised when embedding API returns an error."""
    pass


class EmbeddingService:
    """Service for getting text embeddings via OpenRouter."""
    
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EmbeddingError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable."
            )
        self.model = model
        self.cache_dir = CACHE_DIR / model.replace("/", "_")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _cache_key(self, text: str) -> str:
        """Generate a cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_cached(self, text: str) -> np.ndarray | None:
        """Get cached embedding if it exists."""
        key = self._cache_key(text)
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            return np.load(cache_file)
        return None
    
    def _save_cache(self, text: str, embedding: np.ndarray) -> None:
        """Save embedding to cache."""
        key = self._cache_key(text)
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        # Check cache first
        cached = self._get_cached(text)
        if cached is not None:
            return cached
        
        # Use batch method for single text
        embeddings = self._fetch_embeddings_batch([text])
        embedding = embeddings[0]
        
        # Cache for future use
        self._save_cache(text, embedding)
        
        return embedding
    
    def _fetch_embeddings_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Fetch embeddings for multiple texts in a single API call."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/ankii",
            "X-Title": "Anki Card Clusters",
        }
        
        payload = {
            "model": self.model,
            "input": texts,  # Batch of texts
        }
        
        try:
            response = httpx.post(
                OPENROUTER_URL,
                json=payload,
                headers=headers,
                timeout=120.0  # Longer timeout for batches
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise EmbeddingError(f"OpenRouter API error: {e}")
        
        result = response.json()
        
        if "error" in result:
            raise EmbeddingError(f"OpenRouter error: {result['error']}")
        
        # Extract embeddings in order
        embeddings = [np.array(item["embedding"]) for item in result["data"]]
        return embeddings
    
    def get_embeddings_batch(
        self, 
        texts: list[str], 
        progress_callback: Callable | None = None,
        batch_size: int = 50  # Process 50 texts per API call
    ) -> np.ndarray:
        """Get embeddings for multiple texts with batching.
        
        Args:
            texts: List of texts to embed
            progress_callback: Optional callback(current, total) for progress updates
            batch_size: Number of texts to process per API call
            
        Returns:
            Array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        texts_to_fetch = []
        indices_to_fetch = []
        
        # Check cache for all texts first
        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                all_embeddings.append((i, cached))
            else:
                texts_to_fetch.append(text)
                indices_to_fetch.append(i)
        
        # Fetch uncached texts in batches
        for batch_start in range(0, len(texts_to_fetch), batch_size):
            batch_end = min(batch_start + batch_size, len(texts_to_fetch))
            batch_texts = texts_to_fetch[batch_start:batch_end]
            batch_indices = indices_to_fetch[batch_start:batch_end]
            
            # Fetch batch
            batch_embeddings = self._fetch_embeddings_batch(batch_texts)
            
            # Cache and store results
            for text, idx, emb in zip(batch_texts, batch_indices, batch_embeddings):
                self._save_cache(text, emb)
                all_embeddings.append((idx, emb))
            
            if progress_callback:
                progress_callback(batch_end, len(texts_to_fetch))
        
        # Sort by original index and extract embeddings
        all_embeddings.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in all_embeddings])
