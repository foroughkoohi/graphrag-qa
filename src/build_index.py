"""
Build index embedding for the corpus using bge-small-en-v1.5
"""
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
)

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
INDEX_DIR = PROJECT_ROOT / "cache" / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    print(f"Loading embedding model: {MODEL_NAME}")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f} seconds")

    # Load corpus
    corpus = load_jsonl(PACK / "corpus.jsonl")
    print(f"Number of texts: {len(corpus)}")

    # Extract texts
    doc_ids = [p["doc_id"] for p in corpus]
    texts = [p["text"] for p in corpus]

    # embedding
    print(f"Embedding in progress...")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # for cosine similarity
    )
    print(f"Embedding completed in {time.time() - t0:.1f} seconds")
    print(f"   shape: {embeddings.shape}")

    # Save
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    with open(INDEX_DIR / "doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f)

    print(f"Saved to {INDEX_DIR}")
    print(f"   - embeddings.npy: {embeddings.shape}")
    print(f"   - doc_ids.json: {len(doc_ids)} ids")


if __name__ == "__main__":
    main()