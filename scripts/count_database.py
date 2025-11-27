import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.database import get_db

if __name__ == "__main__":
    print("Connecting to Qdrant...")
    db = get_db()
    print(f"  Total PDF pages in database: {db.count_pdfs()}")
    print(f"  Total images in database: {db.count_images()}")
    print(f"  Total audio files in database: {db.count_audio()}")
