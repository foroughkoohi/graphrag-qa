"""
Dense + Document Mask system
For each question, removes gold_doc_ids from the corpus and retrieves again.
"""
import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
INDEX_DIR = PROJECT_ROOT / "cache" / "index"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = PROJECT_ROOT / "runs" / args.split / "dense_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")

    questions = load_jsonl(PACK / f"questions_{args.split}.jsonl")
    labels = {l["qid"]: l for l in load_jsonl(PACK / f"labels_{args.split}.jsonl")}
    if args.limit:
        questions = questions[:args.limit]
    print(f"Number of questions: {len(questions)}")

    # Load index
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    with open(INDEX_DIR / "doc_ids.json") as f:
        doc_ids = json.load(f)

    doc_id_to_idx = {did: i for i, did in enumerate(doc_ids)}

    # Load embedder
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    # For each question: mask the gold docs and retrieve again
    print(f"\nStarting...")
    t0 = time.time()
    results = []

    for q in tqdm(questions, desc="Dense+Mask"):
        gold = labels[q["qid"]]["gold_doc_ids"]
        gold_indices = {doc_id_to_idx[g] for g in gold if g in doc_id_to_idx}

        # Question embedding
        q_emb = embedder.encode(q["question"], normalize_embeddings=True, convert_to_numpy=True)

        # Score all documents
        scores = embeddings @ q_emb

        # mask: set gold docs to -inf
        for i in gold_indices:
            scores[i] = -np.inf

        # top-k from non-gold documents
        top_idx = np.argsort(-scores)[:args.k]
        retrieved_ids = [doc_ids[i] for i in top_idx]

        results.append({
            "qid": q["qid"],
            "retrieved": retrieved_ids,
            # Note: we don't include the LLM answer for now since the goal is only to evaluate retrieval
            "answer": "",
        })

    total_time = time.time() - t0
    print(f"\n{len(results)} questions in {total_time:.1f} seconds")

    # Save
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"predictions: {pred_path}")


if __name__ == "__main__":
    main()