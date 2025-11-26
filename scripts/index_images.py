#!/usr/bin/env python3
"""Index images into the vector database.

Usage:
    python scripts/index_images.py --data-dir data/images
    python scripts/index_images.py --data-dir data/images --batch-size 32
"""

import argparse
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from backend.config import settings
from backend.database import VectorDB
from backend.models import CLIPEncoder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def find_images(data_dir: Path) -> list[Path]:
    """Find all image files in directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(data_dir.rglob(f"*{ext}"))
        images.extend(data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(images))


def create_payload(image_path: Path, data_dir: Path) -> dict:
    """Create metadata payload for an image."""
    rel_path = image_path.relative_to(data_dir)
    
    # Extract category from subdirectory if present
    parts = rel_path.parts
    category = parts[0] if len(parts) > 1 else None
    
    return {
        "filename": image_path.name,
        "path": f"/images/{rel_path.as_posix()}",
        "category": category,
    }


def index_images(
    data_dir: Path,
    batch_size: int = 16,
    device: str = "cuda:0",
) -> None:
    """Index all images in the data directory."""
    # Find images
    image_paths = find_images(data_dir)
    if not image_paths:
        print(f"No images found in {data_dir}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Initialize encoder and database
    print(f"Loading CLIP model on {device}...")
    encoder = CLIPEncoder(device=device)
    
    print("Connecting to Qdrant...")
    db = VectorDB()
    db.create_image_collection()
    
    # Process in batches
    all_ids = []
    all_embeddings = []
    all_payloads = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding"):
        batch_paths = image_paths[i : i + batch_size]
        
        # Load and encode images
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"\nSkipping {path}: {e}")
        
        if not images:
            continue
        
        # Encode batch
        embeddings = encoder.encode_image(images).numpy()
        
        # Prepare for database
        for path, emb in zip(valid_paths, embeddings):
            image_id = path.stem  # Use filename without extension as ID
            all_ids.append(image_id)
            all_embeddings.append(emb)
            all_payloads.append(create_payload(path, data_dir))
    
    # Insert into database
    print(f"\nInserting {len(all_ids)} embeddings into Qdrant...")
    all_embeddings = np.stack(all_embeddings)
    db.add_images(all_ids, all_embeddings, all_payloads)
    
    print(f"Done! Indexed {len(all_ids)} images.")
    print(f"Total images in collection: {db.count_images()}")


def main():
    parser = argparse.ArgumentParser(description="Index images into vector database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.image_dir,
        help="Directory containing images",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.clip_device,
        help="Device for model inference (cuda:0, cuda:1, cpu)",
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    index_images(args.data_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
