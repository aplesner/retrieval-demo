#!/usr/bin/env python3
"""Test script to check ColPali embedding dimensions."""

from PIL import Image
from backend.models import get_colpali_encoder

def main():
    print("=" * 60)
    print("ColPali Embedding Dimension Test")
    print("=" * 60)

    # Create test image
    print("\n1. Creating test image (224x224 white image)...")
    test_img = Image.new('RGB', (224, 224), color='white')

    # Load encoder
    print("2. Loading ColPali encoder...")
    encoder = get_colpali_encoder()
    print(f"   Model: {encoder.model.__class__.__name__}")
    print(f"   Device: {encoder.device}")

    # Test encoding
    print("\n3. Encoding test image...")
    emb = encoder.encode_pdf_page([test_img])

    # Analyze embedding shape
    print("\n4. Embedding Analysis:")
    print(f"   Shape: {emb.shape}")
    print(f"   - Batch size: {emb.shape[0]}")
    print(f"   - Number of tokens: {emb.shape[1]}")
    print(f"   - Embedding dimension per token: {emb.shape[2]}")

    # Calculate dimensions
    per_token_dim = emb.shape[2]
    num_tokens = emb.shape[1]
    flattened_dim = num_tokens * per_token_dim

    print("\n5. Vector Storage Options:")
    print(f"   Option A (Multi-vector): {per_token_dim} dims per token, {num_tokens} tokens")
    print(f"   Option B (Flattened): {flattened_dim} dims")

    # Check against Qdrant limits
    QDRANT_MAX_DIM = 65536
    print(f"\n6. Qdrant Compatibility:")
    print(f"   Max dimension limit: {QDRANT_MAX_DIM}")
    print(f"   Multi-vector ({per_token_dim}): {'✓ VALID' if per_token_dim <= QDRANT_MAX_DIM else '✗ TOO LARGE'}")
    print(f"   Flattened ({flattened_dim}): {'✓ VALID' if flattened_dim <= QDRANT_MAX_DIM else '✗ TOO LARGE'}")

    # Recommendation
    print("\n7. Recommendation:")
    if flattened_dim > QDRANT_MAX_DIM:
        print(f"   ⚠ Flattened dimension ({flattened_dim}) exceeds Qdrant limit!")
        print(f"   → Must use multi-vector storage with {per_token_dim} dimensions")
    else:
        print(f"   ✓ Both approaches valid, but multi-vector is more accurate for ColPali")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
