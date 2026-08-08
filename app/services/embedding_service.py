from typing import List, Optional
from fastembed import TextEmbedding


class EmbeddingService:
    """Fast & lightweight local embeddings using ONNX Runtime (no OpenAI key required)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        batch_size: int = 256,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.model = TextEmbedding(model_name=self.model_name)

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not text or not text.strip():
            raise ValueError("Text for embedding cannot be empty")
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a list of strings."""
        if not texts:
            return []

        for text in texts:
            if not text or not text.strip():
                raise ValueError("Text for embedding cannot be empty")

        try:
            embeddings_generator = self.model.embed(
                texts, batch_size=self.batch_size
            )
            return [list(emb) for emb in embeddings_generator]
        except Exception as e:
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e