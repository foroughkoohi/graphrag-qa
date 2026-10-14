"""
Document Mask Experiment

Goal: evaluate the stability of the retriever when some gold docs are absent.
Method: for each question, we mask half of the gold docs and see
        how well the retriever can find the other half.

This experiment shows:
- How "stable" the retriever is
- If a gold document is missing, can it find a replacement or not
"""
import json
import os
import sys
import time
from pathlib import Path

# Add the src path to sys.path
sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dense_retriever import DenseRetriever


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
INDEX_DIR = PROJECT_ROOT / "cache" / "index"
OUT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def recall_at_k(retrieved, gold, k):
    """Recall@k = |gold intersect retrieved[:k]| / |gold|"""
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(gold)) / len(gold)


def main():
    print("=" * 70)
    print("Document Mask Experiment")
    print("=" * 70)

    # Load retriever
    retriever = DenseRetriever(INDEX_DIR)

    # Load questions and labels
    questions = load_jsonl(PACK / "questions_dev.jsonl")
    labels = {l["qid"]: l for l in load_jsonl(PACK / "labels_dev.jsonl")}

    # Only the first 100 questions for speed
    N = 100
    questions = questions[:N]
    print(f"\nRunning on the first {len(questions)} dev questions")

    # ===== Store results =====
    results = []
    t0 = time.time()

    for i, q in enumerate(questions, 1):
        qid = q["qid"]
        gold = labels[qid]["gold_doc_ids"]

        # Mask half of the gold docs
        # (for 2 gold docs: mask 1, keep 1)
        # (for 4 gold docs: mask 2, keep 2)
        half = len(gold) // 2
        masked_ids = gold[:half]
        remaining_ids = gold[half:]

        # Retrieve **without** mask (baseline)
        retrieved_no_mask = retriever.retrieve(q["question"], k=10)

        # Retrieve **with** mask (half of gold removed)
        retrieved_with_mask = retriever.retrieve(
            q["question"], k=10, exclude_ids=masked_ids
        )

        # Compute Recall for the remaining gold docs
        # (since masked ones should not be in retrieved_with_mask)
        recall_no_mask = recall_at_k(retrieved_no_mask, remaining_ids, k=10)
        recall_with_mask = recall_at_k(retrieved_with_mask, remaining_ids, k=10)

        # Check that the mask actually worked (masked_ids should not be in with-mask)
        mask_works = not any(mid in retrieved_with_mask for mid in masked_ids)

        results.append({
            "qid": qid,
            "type": q["type"],
            "gold": gold,
            "masked": masked_ids,
            "remaining": remaining_ids,
            "retrieved_no_mask": retrieved_no_mask,
            "retrieved_with_mask": retrieved_with_mask,
            "recall_no_mask": recall_no_mask,
            "recall_with_mask": recall_with_mask,
            "mask_works": mask_works,
        })

        if i % 20 == 0:
            print(f"  {i}/{len(questions)} questions processed")

    elapsed = time.time() - t0
    print(f"\n{len(results)} questions in {elapsed:.1f} seconds")

    # ===== Compute stats =====
    avg_recall_no_mask = sum(r["recall_no_mask"] for r in results) / len(results)
    avg_recall_with_mask = sum(r["recall_with_mask"] for r in results) / len(results)
    mask_correct = sum(1 for r in results if r["mask_works"])

    # ===== Break down by question type =====
    by_type = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r)

    # ===== Print results =====
    print("\n" + "=" * 70)
    print("Overall results")
    print("=" * 70)
    print(f"Recall@10 without mask on remaining: {avg_recall_no_mask:.3f}")
    print(f"Recall@10 with mask on remaining:    {avg_recall_with_mask:.3f}")
    print(f"Performance drop:                    {avg_recall_no_mask - avg_recall_with_mask:+.3f}")
    print(f"Mask worked correctly:               {mask_correct}/{len(results)}")

    print("\n" + "=" * 70)
    print("Breakdown by question type")
    print("=" * 70)
    print(f"{'Type':<20} {'n':>5} {'Recall (no mask)':>18} {'Recall (with mask)':>20} {'Drop':>10}")
    print("-" * 70)

    for t in sorted(by_type):
        type_results = by_type[t]
        n = len(type_results)
        avg_no = sum(r["recall_no_mask"] for r in type_results) / n
        avg_with = sum(r["recall_with_mask"] for r in type_results) / n
        diff = avg_no - avg_with
        print(f"{t:<20} {n:>5} {avg_no:>18.3f} {avg_with:>20.3f} {diff:>+10.3f}")

    # ===== Save results =====
    out_path = OUT_DIR / "mask_experiment.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Document Mask Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Questions tested: {len(results)}\n")
        f.write(f"Time: {elapsed:.1f} seconds\n\n")
        f.write("Overall:\n")
        f.write(f"  Recall@10 (no mask) on remaining:  {avg_recall_no_mask:.4f}\n")
        f.write(f"  Recall@10 (with mask) on remaining: {avg_recall_with_mask:.4f}\n")
        f.write(f"  Drop: {avg_recall_no_mask - avg_recall_with_mask:+.4f}\n")
        f.write(f"  Mask working correctly: {mask_correct}/{len(results)}\n\n")
        f.write("Per question type:\n")
        f.write(f"{'Type':<20} {'n':>5} {'No-Mask R@10':>15} {'With-Mask R@10':>18} {'Drop':>10}\n")
        f.write("-" * 70 + "\n")
        for t in sorted(by_type):
            type_results = by_type[t]
            n = len(type_results)
            avg_no = sum(r["recall_no_mask"] for r in type_results) / n
            avg_with = sum(r["recall_with_mask"] for r in type_results) / n
            f.write(f"{t:<20} {n:>5} {avg_no:>15.4f} {avg_with:>18.4f} {avg_no - avg_with:>+10.4f}\n")

    print(f"\nResults saved: {out_path}")

    # ===== Real examples =====
    print("\n" + "=" * 70)
    print("A few examples")
    print("=" * 70)
    for r in results[:3]:
        print(f"\nqid: {r['qid']} ({r['type']})")
        print(f"  gold:                {r['gold']}")
        print(f"  masked:              {r['masked']}")
        print(f"  remaining:           {r['remaining']}")
        print(f"  retrieved (no mask): {r['retrieved_no_mask'][:5]}")
        print(f"  retrieved (with mask): {r['retrieved_with_mask'][:5]}")
        print(f"  recall no-mask:      {r['recall_no_mask']:.3f}")
        print(f"  recall with-mask:    {r['recall_with_mask']:.3f}")


if __name__ == "__main__":
    main()