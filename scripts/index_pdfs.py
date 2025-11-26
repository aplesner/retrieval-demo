#!/usr/bin/env python3
"""Index PDF documents using ColPali for retrieval."""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
from PIL import Image

from backend.config import settings
from backend.database import get_db
from backend.models import get_colpali_encoder


def index_pdfs(data_dir: Path, recreate: bool = False) -> None:
    """Index PDF files using ColPali.

    Args:
        data_dir: Directory containing PDF files
        recreate: Whether to recreate the collection
    """
    # Initialize
    print("Initializing ColPali encoder...")
    encoder = get_colpali_encoder()

    print("Connecting to Qdrant...")
    db = get_db()

    # Get embedding dimension from a test encoding
    print("Determining embedding dimensions...")
    test_img = Image.new("RGB", (224, 224), color="white")
    test_emb = encoder.encode_pdf_page([test_img])
    embedding_dim = test_emb.shape[1] * test_emb.shape[2]  # num_tokens * embedding_dim_per_token
    print(f"Embedding dimension: {embedding_dim}")

    # Create collection
    if recreate:
        print(f"Recreating collection: {settings.pdf_collection}")
        try:
            db.client.delete_collection(settings.pdf_collection)
        except Exception:
            pass

    db.create_pdf_collection(embedding_dim=embedding_dim)

    # Find PDF files
    pdf_files = list(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files")

    # Create output directory for page images
    output_dir = settings.pdf_image_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index each PDF
    total_pages = 0
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            # Encode PDF pages
            embeddings, page_images = encoder.encode_pdf_from_path(pdf_path)

            # Generate IDs and payloads for each page
            pdf_name = pdf_path.stem
            ids = []
            payloads = []

            for page_num, page_img in enumerate(page_images):
                page_id = f"{pdf_name}_page{page_num}"
                ids.append(page_id)

                # Save page image as thumbnail
                thumb_path = output_dir / f"{page_id}.jpg"
                page_img.save(thumb_path, "JPEG", quality=85, optimize=True)

                payloads.append({
                    "filename": pdf_path.name,
                    "pdf_name": pdf_name,
                    "page": page_num,
                    "page_number": page_num,
                    "path": f"/pdf_images/{page_id}.jpg",
                    "pdf_path": str(pdf_path.relative_to(data_dir.parent)),
                    "total_pages": len(page_images),
                })

            # Add to database
            db.add_pdfs(ids, embeddings, payloads)
            total_pages += len(page_images)

        except Exception as e:
            print(f"\nError processing {pdf_path}: {e}")
            continue

    print(f"\n✓ Indexed {total_pages} pages from {len(pdf_files)} PDFs")
    print(f"  Collection: {settings.pdf_collection}")
    print(f"  Total pages in database: {db.count_pdfs()}")


def main():
    parser = argparse.ArgumentParser(description="Index PDF files for retrieval")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.pdf_dir,
        help=f"Directory containing PDF files (default: {settings.pdf_dir})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the collection (deletes existing data)",
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}")
        sys.exit(1)

    index_pdfs(args.data_dir, recreate=args.recreate)


if __name__ == "__main__":
    main()
