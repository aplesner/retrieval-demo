"""Configuration for the retrieval demo backend."""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # CLIP Model settings (images + text)
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_embedding_dim: int = 512
    clip_device: str = "cuda:0"  # First GPU for CLIP
    
    # CLAP Model settings (audio + text)
    clap_model: str = "laion/clap-htsat-unfused"
    clap_embedding_dim: int = 512
    clap_device: str = "cuda:1"  # Second GPU for CLAP (or same as CLIP if only one)
    
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    image_collection: str = "images"
    audio_collection: str = "audio"
    
    # Data settings
    image_dir: Path = Path("data/images")
    audio_dir: Path = Path("data/audio")
    
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
    
    class Config:
        env_prefix = "RETRIEVAL_"


settings = Settings()
