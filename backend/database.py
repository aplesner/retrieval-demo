"""Qdrant vector database client for images, audio, and PDFs."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MultiVectorConfig,
    MultiVectorComparator,
)
import numpy as np
import torch

from .config import settings


class VectorDB:
    """Qdrant vector database wrapper for multimodal retrieval."""
    
    def __init__(
        self,
        host: str = settings.qdrant_host,
        port: int = settings.qdrant_port,
    ):
        # Set timeout to 300 seconds (5 minutes) to handle large PDF uploads
        self.client = QdrantClient(host=host, port=port, timeout=300)
    
    def create_collection(self, name: str, dim: int) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == name for c in collections)
        
        if not exists:
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
    
    def create_image_collection(self) -> None:
        """Create the images collection."""
        self.create_collection(settings.image_collection, settings.clip_embedding_dim)
    
    def create_audio_collection(self) -> None:
        """Create the audio collection."""
        self.create_collection(settings.audio_collection, settings.clap_embedding_dim)

    def create_pdf_collection(self, embedding_dim: int = 128) -> None:
        """Create the PDF collection with ColPali multi-vector embeddings.

        Args:
            embedding_dim: Dimension of each token embedding (default: 128).
                          Uses multivector configuration with MaxSim comparator.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == settings.pdf_collection for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=settings.pdf_collection,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
    
    def add_items(
        self,
        collection: str,
        ids: list[str],
        embeddings: np.ndarray,
        payloads: list[dict],
    ) -> None:
        """Add embeddings to a collection."""
        points = [
            PointStruct(
                id=idx,
                vector=emb.tolist(),
                payload={"item_id": item_id, **payload},
            )
            for idx, (item_id, emb, payload) in enumerate(zip(ids, embeddings, payloads))
        ]
        
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection,
                points=points[i : i + batch_size],
            )
    
    def add_images(self, ids: list[str], embeddings: np.ndarray, payloads: list[dict]) -> None:
        """Add image embeddings."""
        self.add_items(settings.image_collection, ids, embeddings, payloads)
    
    def add_audio(self, ids: list[str], embeddings: np.ndarray, payloads: list[dict]) -> None:
        """Add audio embeddings."""
        self.add_items(settings.audio_collection, ids, embeddings, payloads)

    def add_pdfs(self, ids: list[str], embeddings: torch.Tensor, payloads: list[dict]) -> None:
        """Add PDF page embeddings (ColPali multi-vector).

        Args:
            ids: List of page IDs (e.g., "doc1_page0", "doc1_page1")
            embeddings: Multi-vector embeddings [num_pages, num_tokens, embedding_dim]
            payloads: Metadata for each page
        """
        # Preserve multi-vector structure for MaxSim scoring
        # Each page has multiple token vectors: [num_tokens, embedding_dim]
        points = []
        for idx, (item_id, emb, payload) in enumerate(zip(ids, embeddings, payloads)):
            # Convert to list of vectors (keeping token-level structure)
            multi_vector = emb.numpy().tolist()  # [num_tokens, embedding_dim]
            # Use hash of item_id to generate unique numeric ID across all PDFs
            # (enumerate idx resets to 0 for each PDF, causing ID collisions!)
            unique_id = hash(item_id) & 0x7FFFFFFF  # Convert to positive 32-bit int
            points.append(
                PointStruct(
                    id=unique_id,
                    vector=multi_vector,  # List of vectors for multivector support
                    payload={"item_id": item_id, **payload},
                )
            )

        # Upsert in smaller batches to avoid timeouts with large PDFs
        # Each page has ~130KB of data (1030 tokens × 128 dims × 4 bytes)
        # 10 pages = ~1.3 MB per batch
        batch_size = 10
        from tqdm import tqdm
        for i in tqdm(range(0, len(points), batch_size), desc="Uploading batches", disable=len(points) <= batch_size):
            self.client.upsert(
                collection_name=settings.pdf_collection,
                points=points[i : i + batch_size],
            )
    
    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = settings.default_limit,
        category: str | None = None,
    ) -> list[dict]:
        """Search for similar items."""
        query_filter = None
        if category:
            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )
        
        results = self.client.query_points(
            collection_name=collection,
            query=query_vector.tolist(),
            limit=limit,
            query_filter=query_filter,
        )
        
        return [
            {
                "id": hit.payload.get("item_id"),
                "score": hit.score,
                "filename": hit.payload.get("filename"),
                "path": hit.payload.get("path"),
                "category": hit.payload.get("category"),
                "duration": hit.payload.get("duration"),  # For audio
            }
            for hit in results.points if hit.payload is not None
        ]
    
    def search_images(self, query_vector: np.ndarray, limit: int = settings.default_limit) -> list[dict]:
        """Search images."""
        return self.search(settings.image_collection, query_vector, limit)
    
    def search_audio(self, query_vector: np.ndarray, limit: int = settings.default_limit) -> list[dict]:
        """Search audio."""
        return self.search(settings.audio_collection, query_vector, limit)

    def search_pdfs(self, query_vector: torch.Tensor, limit: int = settings.default_limit) -> list[dict]:
        """Search PDF pages using ColPali multi-vector embeddings with MaxSim.

        Args:
            query_vector: Multi-vector query embedding [num_tokens, embedding_dim]
            limit: Maximum number of results

        Returns:
            List of matching PDF pages with scores and metadata
        """
        # Preserve multi-vector query structure for MaxSim scoring
        query_multi = query_vector.numpy().tolist()  # [num_tokens, embedding_dim]

        results = self.client.query_points(
            collection_name=settings.pdf_collection,
            query=query_multi,  # Pass as list of vectors
            limit=limit,
        )

        search_results = [
            {
                "id": hit.payload.get("item_id"),
                "score": hit.score,
                "filename": hit.payload.get("filename"),
                "path": hit.payload.get("path"),
                "category": hit.payload.get("category"),
                "page_number": hit.payload.get("page_number", hit.payload.get("page", 0)),
            }
            for hit in results.points if hit.payload is not None
        ]

        return search_results
    
    def get_all_items(self, collection: str, limit: int = settings.max_limit) -> list[dict]:
        """Get all items from a collection."""
        results, _ = self.client.scroll(
            collection_name=collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        
        return [
            {
                "id": point.payload.get("item_id"),
                "filename": point.payload.get("filename"),
                "path": point.payload.get("path"),
                "category": point.payload.get("category"),
                "duration": point.payload.get("duration"),
            }
            for point in results if point.payload is not None
        ]
    
    def get_all_images(self, limit: int = settings.max_limit) -> list[dict]:
        """Get all indexed images."""
        return self.get_all_items(settings.image_collection, limit)
    
    def get_all_audio(self, limit: int = settings.max_limit) -> list[dict]:
        """Get all indexed audio."""
        return self.get_all_items(settings.audio_collection, limit)

    def get_all_pdfs(self, limit: int = settings.max_limit) -> list[dict]:
        """Get all indexed PDF pages."""
        return self.get_all_items(settings.pdf_collection, limit)
    
    def get_embedding(self, collection: str, item_id: str) -> np.ndarray | None:
        """Get the embedding for a specific item."""
        results, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="item_id", match=MatchValue(value=item_id))]
            ),
            limit=1,
            with_vectors=True,
        )
        
        if results:
            return np.array(results[0].vector)
        return None
    
    def get_image_embedding(self, image_id: str) -> np.ndarray | None:
        """Get embedding for a specific image."""
        return self.get_embedding(settings.image_collection, image_id)
    
    def get_audio_embedding(self, audio_id: str) -> np.ndarray | None:
        """Get embedding for a specific audio."""
        return self.get_embedding(settings.audio_collection, audio_id)
    
    def count(self, collection: str) -> int:
        """Get the number of items in a collection."""
        try:
            info = self.client.get_collection(collection)
            return info.points_count
        except Exception:
            return 0
    
    def count_images(self) -> int:
        """Get number of indexed images."""
        return self.count(settings.image_collection)
    
    def count_audio(self) -> int:
        """Get number of indexed audio clips."""
        return self.count(settings.audio_collection)

    def count_pdfs(self) -> int:
        """Get number of indexed PDF pages."""
        return self.count(settings.pdf_collection)


# Singleton instance
_db: VectorDB | None = None


def get_db() -> VectorDB:
    """Get or create the database singleton."""
    global _db
    if _db is None:
        _db = VectorDB()
    return _db
