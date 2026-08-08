import uuid
from typing import Any, Dict, List, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.schemas.document import DocumentChunk


class QdrantVectorStore:
    """Vector store implementation wrapping QdrantClient for chunk index and retrieval."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        location: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "documents",
        vector_size: int = 384,
    ):
        self.collection_name = collection_name
        self.vector_size = vector_size

        if location == ":memory:" or host == ":memory:":
            self.client = QdrantClient(location=":memory:")
        elif location:
            self.client = QdrantClient(url=location, api_key=api_key)
        else:
            self.client = QdrantClient(
                host=host or "qdrant",
                port=port or 6333,
                api_key=api_key,
            )

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it does not exist."""
        collections = [col.name for col in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )

    @staticmethod
    def _to_uuid(chunk_id: str) -> str:
        """Generate deterministic UUID for Qdrant point identity from string chunk_id."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

    @staticmethod
    def _parse_point_data(point: Any) -> Tuple[Dict[str, Any], float]:
        """Extract payload dict and similarity score across in-memory and server Qdrant point structures."""
        if hasattr(point, "payload"):
            payload = point.payload or {}
            score = getattr(point, "score", 0.0)
        elif isinstance(point, tuple):
            first_item = point[0]
            if hasattr(first_item, "payload"):
                payload = first_item.payload or {}
                score = point[1] if len(point) > 1 else getattr(first_item, "score", 0.0)
            else:
                score = point[1] if len(point) > 1 else 0.0
                payload = point[2] if len(point) > 2 and isinstance(point[2], dict) else {}
        elif isinstance(point, dict):
            payload = point.get("payload", {})
            score = point.get("score", 0.0)
        else:
            payload = {}
            score = 0.0

        return payload, score

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        """Store document chunks along with their embeddings and metadata in Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks count and embeddings count must match.")

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            if len(embedding) != self.vector_size:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.vector_size}, got {len(embedding)}"
                )

            payload = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "document_id": chunk.metadata.document_id,
                "chunk_index": chunk.metadata.chunk_index,
                "page_number": chunk.metadata.page_number,
                "source_filename": chunk.metadata.source_filename,
            }

            points.append(
                PointStruct(
                    id=self._to_uuid(chunk.chunk_id),
                    vector=embedding,
                    payload=payload
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

    def search(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search top-k most similar vector chunks using query_points API."""
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit
        )

        results = []
        points = getattr(response, "points", response)
        for point in points:
            payload, score = self._parse_point_data(point)
            results.append({
                "chunk_id": payload.get("chunk_id"),
                "content": payload.get("content"),
                "document_id": payload.get("document_id"),
                "chunk_index": payload.get("chunk_index"),
                "page_number": payload.get("page_number"),
                "source_filename": payload.get("source_filename"),
                "score": score,
            })

        return results