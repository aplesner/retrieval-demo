#!/usr/bin/env python3
"""Download IKEA assembly manual PDFs and convert to images.

Usage:
    python scripts/download_ikea_manuals.py
    python scripts/download_ikea_manuals.py --output-dir data/images/ikea
"""

import argparse
import urllib.request
from pathlib import Path

# IKEA manual PDF URLs from reliable sources
# Format: (name, url)
# Sources: Internet Archive (archive.org) - most reliable
#          IKEA.com direct links (may change)

IKEA_MANUALS = [
    # === INTERNET ARCHIVE (reliable, stable URLs) ===
    
    # MALM series (featured in presentation)
    ("malm-chest-drawers",
     "https://archive.org/download/ikea_malm/ikea_malm.pdf"),
    ("malm-dresser",
     "https://archive.org/download/malm_20220221/malm_20220221.pdf"),
    
    # KALLAX series (popular shelving)
    ("kallax-2x4",
     "https://archive.org/download/ikea-kallax-2x4-assembly-instructions/ikea-kallax-2x4-assembly-instructions.pdf"),
    ("kallax-1x4",
     "https://archive.org/download/ikea-kallax-shelving-unit-assembly-instructions/ikea-kallax-shelving-unit-assembly-instructions.pdf"),
    
    # MICKE desk
    ("micke-desk",
     "https://archive.org/download/ikea-micke-desk-assembly-instructions/ikea-micke-desk-assembly-instructions.pdf"),
    
    # LEKSVIK table
    ("leksvik-table",
     "https://archive.org/download/Ikea.501.160.55/Ikea.501.160.55.pdf"),
    
    # IVAR shelving
    ("ivar-shelf-unit",
     "https://archive.org/download/ikea-AA-2378552-4/ivar-shelf-unit-pine__AA-2378552-4-100.pdf"),
    ("ivar-wall-cabinet",
     "https://archive.org/download/ikea-AA-2272100-5/ikea-AA-2272100-5.pdf"),
    ("ivar-shelf-tall",
     "https://archive.org/download/ikea-aa-2171817-2_202406/ivar-shelf-unit-pine__AA-2171817-2-100.pdf"),
    
    # === IKEA.COM DIRECT (may need verification) ===
    
    # MALM bed (verified working)
    ("malm-bed-frame-low",
     "https://www.ikea.com/us/en/assembly_instructions/malm-bed-frame-low__AA-75286-15_pub.pdf"),
    
    # KALLAX insert
    ("kallax-insert-drawers",
     "https://www.ikea.com/us/en/assembly_instructions/kallax-insert-with-2-drawers-white-stained-oak-effect__AA-1009361-5-100.pdf"),
]

# Additional manuals that can be downloaded from ManualsLib
# These require manual download due to their interface
MANUALSLIB_URLS = {
    "kallax-full": "https://www.manualslib.com/manual/2953875/Ikea-Kallax.html",
    "pax-wardrobe": "https://www.manualslib.com/manual/1147368/Ikea-Pax.html",
    "billy-bookcase": "https://www.manualslib.com/manual/929076/Ikea-Billy.html",
    "hemnes-dresser": "https://www.manualslib.com/manual/929063/Ikea-Hemnes.html",
    "lack-table": "https://www.manualslib.com/manual/929066/Ikea-Lack.html",
    "besta-storage": "https://www.manualslib.com/manual/929054/Ikea-Besta.html",
    "detolf-cabinet": "https://www.manualslib.com/manual/929060/Ikea-Detolf.html",
}


def download_pdf(url: str, output_path: Path) -> bool:
    """Download a PDF file."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(output_path, "wb") as f:
                f.write(response.read())
        return True
    except Exception as e:
        print(f"  Failed to download: {e}")
        return False


def pdf_to_images(pdf_path: Path, output_dir: Path, dpi: int = 150) -> list[Path]:
    """Convert PDF pages to images using pdf2image.
    
    Requires: pip install pdf2image
    And poppler: apt-get install poppler-utils (Linux) or brew install poppler (Mac)
    """
    try:
        from pdf2image import convert_from_path
        
        images = convert_from_path(pdf_path, dpi=dpi)
        output_paths = []
        
        stem = pdf_path.stem
        for i, img in enumerate(images):
            output_path = output_dir / f"{stem}_page{i+1:02d}.jpg"
            img.save(output_path, "JPEG", quality=90)
            output_paths.append(output_path)
        
        return output_paths
    except ImportError:
        print("  pdf2image not installed. Run: pip install pdf2image")
        return []
    except Exception as e:
        print(f"  Failed to convert: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Download IKEA manuals")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "images" / "ikea",
        help="Output directory for images",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "pdfs",
        help="Directory to store downloaded PDFs",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Only download PDFs, don't convert to images",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for image conversion",
    )
    parser.add_argument(
        "--list-additional",
        action="store_true",
        help="List additional manuals available from ManualsLib",
    )
    
    args = parser.parse_args()
    
    if args.list_additional:
        print("Additional manuals available from ManualsLib (manual download required):")
        print("-" * 60)
        for name, url in MANUALSLIB_URLS.items():
            print(f"  {name}: {url}")
        print()
        print("To download: Visit the URL, click 'Download PDF' on each page,")
        print("then save to your pdf-dir and run this script again.")
        return
    
    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pdf_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {len(IKEA_MANUALS)} IKEA manuals...")
    print(f"PDFs: {args.pdf_dir}")
    print(f"Images: {args.output_dir}")
    print()
    
    total_pages = 0
    successful = 0
    failed = []
    
    for name, url in IKEA_MANUALS:
        pdf_path = args.pdf_dir / f"{name}.pdf"
        
        # Download PDF
        if not pdf_path.exists():
            print(f"Downloading {name}...")
            if not download_pdf(url, pdf_path):
                failed.append(name)
                continue
            successful += 1
        else:
            print(f"Skipping {name} (already downloaded)")
            successful += 1
        
        # Convert to images
        if not args.skip_convert:
            print(f"  Converting to images...")
            pages = pdf_to_images(pdf_path, args.output_dir, dpi=args.dpi)
            total_pages += len(pages)
            print(f"  Created {len(pages)} page images")
    
    print()
    print(f"Done! Successfully processed: {successful}/{len(IKEA_MANUALS)}")
    if failed:
        print(f"Failed downloads: {', '.join(failed)}")
    print(f"Total pages: {total_pages}")
    print(f"Images saved to: {args.output_dir}")
    print()
    print("For more manuals, run: python scripts/download_ikea_manuals.py --list-additional")


if __name__ == "__main__":
    main()
