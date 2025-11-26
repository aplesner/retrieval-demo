#!/usr/bin/env python3
"""Download sample datasets for the retrieval demo.

Usage:
    python scripts/download_data.py --dataset coco-val2017
    python scripts/download_data.py --dataset sample-images
    python scripts/download_data.py --list
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "images"


DATASETS = {
    "coco-val2017": {
        "description": "COCO 2017 validation set (5,000 images, ~1GB)",
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "size": "~1GB",
    },
    "sample-images": {
        "description": "Small sample of placeholder images for testing",
        "urls": [
            ("dog.jpg", "https://picsum.photos/seed/dog/800/600"),
            ("cat.jpg", "https://picsum.photos/seed/cat/800/600"),
            ("city.jpg", "https://picsum.photos/seed/city/800/600"),
            ("nature.jpg", "https://picsum.photos/seed/nature/800/600"),
            ("food.jpg", "https://picsum.photos/seed/food/800/600"),
            ("car.jpg", "https://picsum.photos/seed/car/800/600"),
            ("beach.jpg", "https://picsum.photos/seed/beach/800/600"),
            ("mountain.jpg", "https://picsum.photos/seed/mountain/800/600"),
            ("office.jpg", "https://picsum.photos/seed/office/800/600"),
            ("park.jpg", "https://picsum.photos/seed/park/800/600"),
        ]
    },
    "unsplash-sample": {
        "description": "Sample from Unsplash Lite dataset (manual download)",
        "url": None,
        "instructions": """
To download Unsplash Lite:
1. Go to https://unsplash.com/data
2. Download the Lite dataset
3. Extract photos to data/images/unsplash/
"""
    },
}


def download_file(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Download a file with progress."""
    try:
        print(f"{desc}...")
        print(f"  URL: {url}")
        print(f"  Destination: {output_path}")
        
        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 // total_size)
                mb_done = count * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r  Progress: {percent}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, output_path, progress_hook)
        print()
        return True
    except Exception as e:
        print(f"\n  Error: {e}")
        return False


def download_coco_val2017(output_dir: Path) -> None:
    """Download and extract COCO val2017 dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = DATASETS["coco-val2017"]
    zip_path = output_dir.parent / "coco_val2017.zip"
    extract_dir = output_dir / "coco"
    
    # Check if already extracted
    if extract_dir.exists() and any(extract_dir.iterdir()):
        count = sum(1 for _ in extract_dir.rglob("*.jpg"))
        print(f"COCO val2017 already exists at {extract_dir} ({count} images)")
        return
    
    # Download
    if not zip_path.exists():
        print(f"Downloading COCO val2017 ({dataset['size']})...")
        if not download_file(dataset["url"], zip_path, "Downloading COCO val2017"):
            return
    else:
        print(f"Using cached zip: {zip_path}")
    
    # Extract
    print("Extracting...")
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Extract val2017 folder contents directly to coco/
        for member in z.namelist():
            if member.startswith("val2017/") and member.endswith(".jpg"):
                # Extract to coco/ directory
                filename = Path(member).name
                target = extract_dir / filename
                with z.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())
    
    count = sum(1 for _ in extract_dir.glob("*.jpg"))
    print(f"\nDone! Extracted {count} images to {extract_dir}")
    
    # Optionally remove zip to save space
    print(f"Tip: Remove {zip_path} to save ~1GB disk space")


def download_sample_images(output_dir: Path) -> None:
    """Download sample placeholder images for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = DATASETS["sample-images"]
    
    for filename, url in dataset["urls"]:
        output_path = output_dir / filename
        if output_path.exists():
            print(f"  Skipping {filename} (exists)")
            continue
        
        print(f"  Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, output_path)
        except Exception as e:
            print(f"    Failed: {e}")
    
    print(f"\nDownloaded {len(dataset['urls'])} sample images to {output_dir}")


def list_datasets() -> None:
    """List all available datasets."""
    print("Available datasets:\n")
    for key, info in DATASETS.items():
        print(f"  {key}:")
        print(f"    {info['description']}")
        if info.get('url'):
            print(f"    Size: {info.get('size', 'Unknown')}")
        elif info.get('urls'):
            print(f"    Files: {len(info['urls'])}")
        else:
            print(f"    Note: Manual download required")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download datasets for retrieval demo")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="coco-val2017",
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
        list_datasets()
        return
    
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}\n")
    
    if args.dataset == "coco-val2017":
        download_coco_val2017(args.output_dir)
    elif args.dataset == "sample-images":
        download_sample_images(args.output_dir)
    else:
        dataset = DATASETS[args.dataset]
        if dataset.get("instructions"):
            print(f"Description: {dataset['description']}")
            print(dataset["instructions"])
        else:
            print(f"Download not implemented for {args.dataset}")


if __name__ == "__main__":
    main()
