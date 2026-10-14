"""
Test dense retrieval on a few questions
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
)

import numpy as np
from sentence_transformers import SentenceTransformer


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
INDEX_DIR = PROJECT_ROOT / "cache" / "index"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    # Load index
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    with open(INDEX_DIR / "doc_ids.json", encoding="utf-8") as f:
        doc_ids = json.load(f)

    # Load model
    model = SentenceTransformer(MODEL_NAME)

    # Load a few questions
    questions = load_jsonl(PACK / "questions_dev.jsonl")[:5]
    labels = {l["qid"]: l for l in load_jsonl(PACK / "labels_dev.jsonl")}

    print("=" * 70)
    for q in questions:
        # Question embedding
        q_emb = model.encode(
            q["question"],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Cosine similarity (since we normalized, dot product = cosine)
        scores = embeddings @ q_emb

        # top-10
        top_idx = np.argsort(-scores)[:10]
        top_ids = [doc_ids[i] for i in top_idx]
        top_scores = [scores[i] for i in top_idx]

        # gold
        gold = labels[q["qid"]]["gold_doc_ids"]

        print(f"\n[{q['type']}] {q['qid']}: {q['question']}")
        print(f"  Gold: {gold}")
        print(f"  Top-10: {top_ids}")
        print(f"  Scores: {[f'{s:.3f}' for s in top_scores]}")

        # Check where the gold docs are
        for g in gold:
            if g in top_ids:
                rank = top_ids.index(g) + 1
                print(f"  [HIT] {g} at rank {rank}")
            else:
                print(f"  [MISS] {g} not in top-10")


if __name__ == "__main__":
    main()