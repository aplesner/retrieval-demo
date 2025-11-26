#!/usr/bin/env python3
"""Download audio datasets for the retrieval demo.

Usage:
    python scripts/download_audio.py --dataset esc50
    python scripts/download_audio.py --dataset esc50 --output-dir data/audio
"""

import argparse
import os
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "audio"


DATASETS = {
    "esc50": {
        "name": "ESC-50: Environmental Sound Classification",
        "description": "2,000 environmental audio recordings (5 seconds each), 50 classes",
        "url": "https://github.com/karoldvl/ESC-50/archive/master.zip",
        "size": "~600MB",
        "classes": [
            "dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects",
            "sheep", "crow", "rain", "sea_waves", "crackling_fire", "crickets",
            "chirping_birds", "water_drops", "wind", "pouring_water", "toilet_flush",
            "thunderstorm", "crying_baby", "sneezing", "clapping", "breathing",
            "coughing", "footsteps", "laughing", "brushing_teeth", "snoring",
            "drinking_sipping", "door_wood_knock", "mouse_click", "keyboard_typing",
            "door_wood_creaks", "can_opening", "washing_machine", "vacuum_cleaner",
            "clock_alarm", "clock_tick", "glass_breaking", "helicopter", "chainsaw",
            "siren", "car_horn", "engine", "train", "church_bells", "airplane",
            "fireworks", "hand_saw"
        ]
    },
    "urbansound8k": {
        "name": "UrbanSound8K",
        "description": "8,732 labeled sound excerpts of urban sounds",
        "url": None,  # Requires registration
        "instructions": """
UrbanSound8K requires registration to download:
1. Go to https://urbansounddataset.weebly.com/urbansound8k.html
2. Fill out the form and download the dataset
3. Extract to data/audio/urbansound8k/
"""
    },
    "fsd50k": {
        "name": "FSD50K",
        "description": "51,197 audio clips from Freesound",
        "url": None,
        "instructions": """
FSD50K can be downloaded from Zenodo:
1. Go to https://zenodo.org/record/4060432
2. Download the dataset files
3. Extract to data/audio/fsd50k/
"""
    }
}


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress."""
    try:
        print(f"{desc}...")
        
        def progress_hook(count, block_size, total_size):
            percent = min(100, count * block_size * 100 // total_size)
            print(f"\r  Progress: {percent}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_esc50(output_dir: Path) -> None:
    """Download and extract ESC-50 dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = DATASETS["esc50"]
    zip_path = output_dir / "esc50.zip"
    
    # Download
    if not zip_path.exists():
        print(f"Downloading ESC-50 ({dataset['size']})...")
        if not download_file(dataset["url"], zip_path, "Downloading ESC-50"):
            return
    else:
        print("ESC-50 zip already downloaded")
    
    # Extract
    extract_dir = output_dir / "ESC-50-master"
    if not extract_dir.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
    
    # Organize by category
    audio_src = extract_dir / "audio"
    meta_file = extract_dir / "meta" / "esc50.csv"
    
    if audio_src.exists() and meta_file.exists():
        print("Organizing audio by category...")
        
        # Read metadata
        import csv
        with open(meta_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row['filename']
                category = row['category']
                
                src_path = audio_src / filename
                if src_path.exists():
                    # Create category directory
                    cat_dir = output_dir / category.replace(' ', '_')
                    cat_dir.mkdir(exist_ok=True)
                    
                    # Copy file
                    dst_path = cat_dir / filename
                    if not dst_path.exists():
                        shutil.copy2(src_path, dst_path)
        
        print(f"Organized audio into {len(dataset['classes'])} categories")
    
    # Cleanup (optional)
    # shutil.rmtree(extract_dir)
    # zip_path.unlink()
    
    print(f"\nESC-50 ready at: {output_dir}")
    print(f"Total audio files: {sum(1 for _ in output_dir.rglob('*.wav'))}")


def main():
    parser = argparse.ArgumentParser(description="Download audio datasets")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="esc50",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets",
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("Available audio datasets:\n")
        for key, info in DATASETS.items():
            print(f"  {key}:")
            print(f"    Name: {info['name']}")
            print(f"    Description: {info['description']}")
            if info.get('url'):
                print(f"    Size: {info.get('size', 'Unknown')}")
            else:
                print(f"    Note: Requires manual download")
            print()
        return
    
    dataset = DATASETS[args.dataset]
    
    print(f"Dataset: {dataset['name']}")
    print(f"Description: {dataset['description']}")
    print(f"Output: {args.output_dir}\n")
    
    if args.dataset == "esc50":
        download_esc50(args.output_dir)
    elif dataset.get("instructions"):
        print(dataset["instructions"])
    else:
        print("Dataset download not implemented")


if __name__ == "__main__":
    main()
