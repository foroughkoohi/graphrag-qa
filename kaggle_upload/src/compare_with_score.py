"""
Compare our metrics with the instructor's score.py
"""
import json
import random
import subprocess
import sys
from pathlib import Path

# Add src path to sys.path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import evaluate_one


PACK = Path("data/course_pack_v1")
OUT_DIR = Path("runs/dev/_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_dummy_predictions(seed=42):
    """
    Creates a random predictions.jsonl file.
    """
    corpus = load_jsonl(PACK / "corpus.jsonl")
    questions = load_jsonl(PACK / "questions_dev.jsonl")
    labels = load_jsonl(PACK / "labels_dev.jsonl")
    label_map = {lab["qid"]: lab for lab in labels}

    doc_ids = [d["doc_id"] for d in corpus]
    rng = random.Random(seed)

    # Possible answer pool for diversity
    answer_pool = ["yes", "no", "John Carpenter", "The Third Side Of The River",
                   "English", "1948", "Huntington's disease", "American"]

    predictions = []
    for q in questions:
        # Random retrieval: 10 random doc_ids
        retrieved = rng.sample(doc_ids, 10)
        # Random answer
        answer = rng.choice(answer_pool)
        predictions.append({
            "qid": q["qid"],
            "answer": answer,
            "retrieved": retrieved,
        })

    out_path = OUT_DIR / "predictions.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"{len(predictions)} predictions written to {out_path}")
    return predictions, label_map


def run_score_py():
    """
    Runs the instructor's score.py and returns the output.
    """
    cmd = [
        sys.executable, "score.py",
        str(OUT_DIR / "predictions.jsonl"),
        "--pack", str(PACK),
        "--split", "dev",
    ]
    # score.py is in the project root
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("score.py encountered an error:")
        print(result.stderr)
        print(result.stdout)
        sys.exit(1)
    return result.stdout


def parse_score_output(output):
    """
    Extracts overall metrics from score.py output.
    """
    metrics = {}
    in_overall = False
    for line in output.splitlines():
        if line.strip().startswith("overall"):
            in_overall = True
            continue
        if in_overall:
            if line.strip() == "" or line.strip().startswith("type:"):
                break
            # Line like: "  recall@2         45.3   [ 43.2, 47.5]"
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                try:
                    value = float(parts[1]) / 100.0  # Convert percentage to ratio
                    metrics[name] = value
                except ValueError:
                    pass
    return metrics


def my_score(predictions, label_map):
    """
    Calculates our metrics over all questions.
    """
    all_metrics = {}
    for p in predictions:
        lab = label_map[p["qid"]]
        m = evaluate_one(p, lab)
        for k, v in m.items():
            all_metrics.setdefault(k, []).append(v)

    # Average
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}


def compare(my, theirs):
    """
    Compare two metric dictionaries.
    """
    print("\n" + "=" * 70)
    print(f"{'metric':<18} {'mine':>10} {'score.py':>10} {'diff':>10}")
    print("=" * 70)

    all_keys = sorted(set(my.keys()) | set(theirs.keys()))
    for key in all_keys:
        mine = my.get(key, None)
        their = theirs.get(key, None)

        if mine is None:
            print(f"{key:<18} {'—':>10} {their:>10.4f} {'—':>10}")
        elif their is None:
            print(f"{key:<18} {mine:>10.4f} {'—':>10} {'—':>10}")
        else:
            diff = mine - their
            flag = "OK" if abs(diff) < 0.005 else "WARN"
            print(f"{key:<18} {mine:>10.4f} {their:>10.4f} {diff:>+10.4f}  {flag}")


if __name__ == "__main__":
    print("Generating random predictions...")
    predictions, label_map = make_dummy_predictions(seed=42)

    print("\nCalculating with our metrics...")
    my = my_score(predictions, label_map)
    for k, v in sorted(my.items()):
        print(f"  {k:<15} {v:.4f}")

    print("\nRunning instructor's score.py...")
    out = run_score_py()
    print(out)

    print("\nExtracting score.py metrics...")
    theirs = parse_score_output(out)
    for k, v in sorted(theirs.items()):
        print(f"  {k:<15} {v:.4f}")

    print("\nComparing results...")
    compare(my, theirs)