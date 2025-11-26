"""FastAPI backend for multimodal retrieval demo."""

from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings
from .database import get_db
from .models import get_clip_encoder, get_clap_encoder


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and database on startup."""
    # Load CLIP model
    print(f"Loading CLIP model: {settings.clip_model}")
    get_clip_encoder()
    print("CLIP model loaded.")
    
    # Load CLAP model
    print(f"Loading CLAP model: {settings.clap_model}")
    try:
        get_clap_encoder()
        print("CLAP model loaded.")
    except Exception as e:
        print(f"Warning: CLAP model failed to load: {e}")
        print("Audio search will be unavailable.")
    
    # Initialize database connection
    db = get_db()
    db.create_image_collection()
    db.create_audio_collection()
    print(f"Connected to Qdrant. Collections ready.")
    print(f"  Images: {db.count_images()}")
    print(f"  Audio: {db.count_audio()}")
    
    yield
    
    print("Shutting down...")


app = FastAPI(
    title="Multimodal Retrieval Demo",
    description="Search images and audio using text, images, or audio clips",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request/Response Models
# =============================================================================

class TextSearchRequest(BaseModel):
    query: str
    limit: int = settings.default_limit


class ImageSearchRequest(BaseModel):
    image_id: str
    limit: int = settings.default_limit


class AudioSearchRequest(BaseModel):
    audio_id: str
    limit: int = settings.default_limit


class MultimodalSearchRequest(BaseModel):
    text_query: str
    image_id: str
    text_weight: float = 0.5
    limit: int = settings.default_limit


class SearchResult(BaseModel):
    id: str
    score: float
    filename: str
    path: str
    category: str | None = None
    duration: float | None = None  # For audio


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_type: str


class ItemInfo(BaseModel):
    id: str
    filename: str
    path: str
    category: str | None = None
    duration: float | None = None


# =============================================================================
# Health & Info Endpoints
# =============================================================================

@app.get("/api/health")
async def health_check() -> dict:
    """Health check endpoint."""
    db = get_db()
    return {
        "status": "healthy",
        "clip_model": settings.clip_model,
        "clap_model": settings.clap_model,
        "indexed_images": db.count_images(),
        "indexed_audio": db.count_audio(),
    }


@app.get("/api/suggestions")
async def get_suggestions() -> dict:
    """Get suggested search queries."""
    return {
        "image_suggestions": settings.image_suggestions,
        "audio_suggestions": settings.audio_suggestions,
    }


# =============================================================================
# Image Search Endpoints
# =============================================================================

@app.post("/api/search/text", response_model=SearchResponse)
async def search_images_by_text(request: TextSearchRequest) -> SearchResponse:
    """Search images using a text query."""
    encoder = get_clip_encoder()
    db = get_db()
    
    embedding = encoder.encode_text([request.query])
    query_vector = embedding[0].numpy()
    
    results = db.search_images(query_vector, limit=min(request.limit, settings.max_limit))
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type="text",
    )


@app.post("/api/search/image", response_model=SearchResponse)
async def search_images_by_image(request: ImageSearchRequest) -> SearchResponse:
    """Search images using another image."""
    db = get_db()
    
    query_vector = db.get_image_embedding(request.image_id)
    if query_vector is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_id}")
    
    results = db.search_images(query_vector, limit=min(request.limit, settings.max_limit) + 1)
    results = [r for r in results if r["id"] != request.image_id][:request.limit]
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type="image",
    )


@app.post("/api/search/multimodal", response_model=SearchResponse)
async def search_images_multimodal(request: MultimodalSearchRequest) -> SearchResponse:
    """Search images using both text and image."""
    encoder = get_clip_encoder()
    db = get_db()
    
    text_embedding = encoder.encode_text([request.text_query])[0].numpy()
    
    image_embedding = db.get_image_embedding(request.image_id)
    if image_embedding is None:
        raise HTTPException(status_code=404, detail=f"Image not found: {request.image_id}")
    
    w = np.clip(request.text_weight, 0.0, 1.0)
    combined = w * text_embedding + (1 - w) * image_embedding
    combined = combined / np.linalg.norm(combined)
    
    results = db.search_images(combined, limit=min(request.limit, settings.max_limit) + 1)
    results = [r for r in results if r["id"] != request.image_id][:request.limit]
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type="multimodal",
    )


@app.get("/api/images", response_model=list[ItemInfo])
async def list_images(limit: int = settings.default_limit) -> list[ItemInfo]:
    """List all indexed images."""
    db = get_db()
    images = db.get_all_images(limit=min(limit, settings.max_limit))
    return [ItemInfo(**img) for img in images]


# =============================================================================
# Audio Search Endpoints
# =============================================================================

@app.post("/api/search/audio/text", response_model=SearchResponse)
async def search_audio_by_text(request: TextSearchRequest) -> SearchResponse:
    """Search audio using a text query."""
    encoder = get_clap_encoder()
    db = get_db()
    
    embedding = encoder.encode_text([request.query])
    query_vector = embedding[0].numpy()
    
    results = db.search_audio(query_vector, limit=min(request.limit, settings.max_limit))
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type="text_to_audio",
    )


@app.post("/api/search/audio/audio", response_model=SearchResponse)
async def search_audio_by_audio(request: AudioSearchRequest) -> SearchResponse:
    """Search audio using another audio clip."""
    db = get_db()
    
    query_vector = db.get_audio_embedding(request.audio_id)
    if query_vector is None:
        raise HTTPException(status_code=404, detail=f"Audio not found: {request.audio_id}")
    
    results = db.search_audio(query_vector, limit=min(request.limit, settings.max_limit) + 1)
    results = [r for r in results if r["id"] != request.audio_id][:request.limit]
    
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query_type="audio_to_audio",
    )


@app.get("/api/audio", response_model=list[ItemInfo])
async def list_audio(limit: int = settings.default_limit) -> list[ItemInfo]:
    """List all indexed audio clips."""
    db = get_db()
    audio = db.get_all_audio(limit=min(limit, settings.max_limit))
    return [ItemInfo(**a) for a in audio]


# =============================================================================
# Static Files
# =============================================================================

if settings.image_dir.exists():
    app.mount("/images", StaticFiles(directory=settings.image_dir), name="images")

if settings.audio_dir.exists():
    app.mount("/audio", StaticFiles(directory=settings.audio_dir), name="audio")
