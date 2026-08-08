from typing import List, Optional
from openai import OpenAI


class EmbeddingService:
    """Service for generating text embeddings using OpenAI API with batching support."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
        batch_size: int = 200,
    ):
        if not api_key:
            raise ValueError("API key for OpenAI is required")

        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        vectors = self.embed_batch([text])
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of strings with sub-batching."""
        if not texts:
            return []

        # Validation of each element
        for text in texts:
            if not text or not text.strip():
                raise ValueError("Text for embedding cannot be empty")

        all_embeddings: List[List[float]] = []

        # Split into sub-batches based on batch_size
        for i in range(0, len(texts), self.batch_size):
            chunk_batch = texts[i : i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk_batch
                )
            except Exception as e:
                raise RuntimeError(f"Failed to generate embeddings: {e}") from e

            for item in response.data:
                embedding = item.embedding
                if len(embedding) != self.dimension:
                    raise ValueError(
                        f"Returned embedding dimension mismatch: expected {self.dimension}, got {len(embedding)}"
                    )
                all_embeddings.append(embedding)

        return all_embeddings