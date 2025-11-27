"""Configuration for the retrieval demo backend."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # CLIP Model settings (images + text)
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embedding_dim: int = 512
    clip_device: str = "cuda:1"  # GPU 1 with CLAP (smaller models)

    # CLAP Model settings (audio + text)
    clap_model: str = "laion/clap-htsat-unfused"
    clap_embedding_dim: int = 512
    clap_device: str = "cuda:0"  # GPU 1 with CLIP (smaller models)

    # ColPali Model settings (PDF + text)
    colpali_model: str = "vidore/colpali-v1.2"
    colpali_embedding_dim: int = 128  # ColPali uses multi-vector embeddings
    colpali_device: str = "cuda:0"  # GPU for ColPali

    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    image_collection: str = "images"
    audio_collection: str = "audio"
    pdf_collection: str = "pdfs"

    # Data settings
    image_dir: Path = Path("data/images")
    audio_dir: Path = Path("data/audio")
    pdf_dir: Path = Path("data/pdfs")
    pdf_image_dir: Path = Path("data/pdf_images")  # Rendered page thumbnails
    
    # Search settings
    default_limit: int = 50
    max_limit: int = 100
    
    # Suggested queries for the demo
    image_suggestions: list[str] = [
        "a dog playing in the park",
        "sunset over the ocean",
        "people working in an office",
        "a red sports car",
        "food on a plate",
        "mountains with snow",
    ]
    
    audio_suggestions: list[str] = [
        "dog barking",
        "rain falling on a roof",
        "car engine starting",
        "birds singing in the morning",
        "people talking in a cafe",
        "piano music",
        "thunder and lightning",
        "ocean waves crashing",
    ]

    pdf_suggestions: list[str] = [
        "how to assemble a drawer",
        "installation instructions for shelves",
        "tools required for assembly",
        "screw and bolt specifications",
        "safety warnings and precautions",
        "exploded view diagram",
    ]
    
    class Config:
        env_prefix = "RETRIEVAL_"


settings = Settings()
