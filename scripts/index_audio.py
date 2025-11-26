#!/usr/bin/env python3
"""Index audio files into the vector database.

Usage:
    python scripts/index_audio.py --data-dir data/audio
    python scripts/index_audio.py --data-dir data/audio --batch-size 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import librosa
import numpy as np
from tqdm import tqdm

from backend.config import settings
from backend.database import VectorDB
from backend.models import CLAPEncoder


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def find_audio_files(data_dir: Path) -> list[Path]:
    """Find all audio files in directory."""
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(data_dir.rglob(f"*{ext}"))
        files.extend(data_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def get_audio_duration(path: Path) -> float:
    """Get duration of audio file in seconds."""
    try:
        return librosa.get_duration(path=path)
    except Exception:
        return 0.0


def create_payload(audio_path: Path, data_dir: Path) -> dict:
    """Create metadata payload for an audio file."""
    rel_path = audio_path.relative_to(data_dir)
    
    # Extract category from subdirectory if present
    parts = rel_path.parts
    category = parts[0] if len(parts) > 1 else None
    
    duration = get_audio_duration(audio_path)
    
    return {
        "filename": audio_path.name,
        "path": f"/audio/{rel_path.as_posix()}",
        "category": category,
        "duration": round(duration, 2),
    }


def index_audio(
    data_dir: Path,
    batch_size: int = 8,
    device: str = "cuda:1",
) -> None:
    """Index all audio files in the data directory."""
    # Find audio files
    audio_paths = find_audio_files(data_dir)
    if not audio_paths:
        print(f"No audio files found in {data_dir}")
        return
    
    print(f"Found {len(audio_paths)} audio files")
    
    # Initialize encoder and database
    print(f"Loading CLAP model on {device}...")
    encoder = CLAPEncoder(device=device)
    
    print("Connecting to Qdrant...")
    db = VectorDB()
    db.create_audio_collection()
    
    # Process in batches
    all_ids = []
    all_embeddings = []
    all_payloads = []
    
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding"):
        batch_paths = audio_paths[i : i + batch_size]
        
        # Load and encode audio
        audio_arrays = []
        sample_rates = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                audio, sr = librosa.load(path, sr=None, mono=True)
                audio_arrays.append(audio)
                sample_rates.append(sr)
                valid_paths.append(path)
            except Exception as e:
                print(f"\nSkipping {path}: {e}")
        
        if not audio_arrays:
            continue
        
        # Encode batch
        embeddings = encoder.encode_audio(audio_arrays, sample_rates).numpy()
        
        # Prepare for database
        for path, emb in zip(valid_paths, embeddings):
            audio_id = path.stem
            all_ids.append(audio_id)
            all_embeddings.append(emb)
            all_payloads.append(create_payload(path, data_dir))
    
    # Insert into database
    print(f"\nInserting {len(all_ids)} embeddings into Qdrant...")
    all_embeddings = np.stack(all_embeddings)
    db.add_audio(all_ids, all_embeddings, all_payloads)
    
    print(f"Done! Indexed {len(all_ids)} audio files.")
    print(f"Total audio in collection: {db.count_audio()}")


def main():
    parser = argparse.ArgumentParser(description="Index audio into vector database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.audio_dir,
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for encoding (smaller for audio due to memory)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=settings.clap_device,
        help="Device for model inference",
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Data directory does not exist: {args.data_dir}")
        sys.exit(1)
    
    index_audio(args.data_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
