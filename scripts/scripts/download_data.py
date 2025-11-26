#!/usr/bin/env python3
"""Download sample datasets for the retrieval demo.

Usage:
    python scripts/download_data.py --dataset coco-sample
    python scripts/download_data.py --dataset unsplash-sample
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "images"


DATASETS = {
    "coco-sample": {
        "description": "COCO 2017 validation subset (~1000 images)",
        "url": None,  # Would need to implement COCO download
        "instructions": """
To download COCO images:
1. Go to https://cocodataset.org/#download
2. Download "2017 Val images [5K/1GB]"
3. Extract to data/images/coco/

Or use fiftyone:
    pip install fiftyone
    python -c "import fiftyone.zoo as foz; foz.load_zoo_dataset('coco-2017', split='validation', max_samples=1000)"
"""
    },
    "unsplash-sample": {
        "description": "Sample from Unsplash Lite dataset",
        "url": None,
        "instructions": """
To download Unsplash Lite:
1. Go to https://unsplash.com/data
2. Download the Lite dataset
3. Extract photos to data/images/unsplash/
"""
    },
    "sample-images": {
        "description": "Small sample of Creative Commons images for testing",
        "urls": [
            # Using Picsum for placeholder images (replace with actual CC images for production)
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
    }
}


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


def main():
    parser = argparse.ArgumentParser(description="Download datasets for retrieval demo")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="sample-images",
        help="Dataset to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}\n")
    
    if args.dataset == "sample-images":
        download_sample_images(args.output_dir)
    else:
        dataset = DATASETS[args.dataset]
        print(f"Description: {dataset['description']}")
        print(dataset["instructions"])


if __name__ == "__main__":
    main()
