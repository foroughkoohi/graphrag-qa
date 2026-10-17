"""
Error analysis for Phase 1 report.
Finds 6 categories of errors and prints full details for each.
Run from project root: python find_errors.py
"""
import json
import re
import string
from collections import Counter
from pathlib import Path

# ============ CONFIG ============
PACK = Path("data/course_pack_v1")
RUNS = Path("runs_qwen/dev")
OUT_DIR = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_PER_CATEGORY = 2
# ================================

# ============ LOAD ============
questions = {q["qid"]: q for q in (
    json.loads(l) for l in open(PACK / "questions_dev.jsonl", encoding="utf-8") if l.strip()
)}
labels = {l["qid"]: l for l in (
    json.loads(l) for l in open(PACK / "labels_dev.jsonl", encoding="utf-8") if l.strip()
)}
corpus = {p["doc_id"]: p for p in (
    json.loads(l) for l in open(PACK / "corpus.jsonl", encoding="utf-8") if l.strip()
)}

def load_preds(sys_name):
    path = RUNS / sys_name / "predictions.jsonl"
    return {p["qid"]: p for p in (
        json.loads(l) for l in open(path, encoding="utf-8") if l.strip()
    )}

closed = load_preds("closed_book")
dense = load_preds("dense")
oracle = load_preds("oracle")

# ============ METRICS ============
def normalize(s):
    s = "".join(c for c in s.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())

def em(pred, gold_list):
    return int(normalize(pred) in {normalize(g) for g in gold_list})

def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    c = sum((Counter(p) & Counter(g)).values())
    if c == 0:
        return 0.0
    pr, rc = c / len(p), c / len(g)
    return 2 * pr * rc / (pr + rc)

def max_f1(pred, gold_list):
    return max(f1(pred, g) for g in gold_list) if pred else 0.0

def recall_at_k(retrieved, gold, k):
    if not gold:
        return 0.0
    return len(set(retrieved[:k]) & set(gold)) / len(gold)

def hitall_at_k(retrieved, gold, k):
    if not gold:
        return 0.0
    return float(set(gold) <= set(retrieved[:k]))

# ============ FORMAT HELPERS ============
def format_text(t, width=70):
    """Wrap text for readable output."""
    t = " ".join(t.split())
    return t

def print_sample_separator(n):
    print()
    print("┌" + "─" * 78 + "┐")
    print(f"│ SAMPLE {n}" + " " * 71 + "│")
    print("└" + "─" * 78 + "┘")
    print()

def print_sample(qid, question, pred_answer, gold_answer, gold_docs, retrieved, extra_lines=None):
    print(f"  qid:          {qid}")
    print(f"  type:         {questions[qid]['type']}")
    print(f"  question:     {format_text(question)}")
    print(f"  system:       {pred_answer!r}")
    print(f"  gold answer:  {gold_answer!r}")
    print(f"  gold doc ids: {gold_docs}")
    print(f"  retrieved:    {retrieved}")
    if extra_lines:
        for label, value in extra_lines:
            print(f"  {label:14s}{value}")
    print()

def print_passage_snippet(doc_id, n_chars=250):
    text = format_text(corpus[doc_id]["text"])
    print(f"    [{doc_id}] {text[:n_chars]}{'...' if len(text) > n_chars else ''}")

# ============ RUNNER ============
def find_samples(predicate, n=SAMPLES_PER_CATEGORY):
    """Return first N qids where predicate(qid, label) is True."""
    found = []
    for qid, lab in labels.items():
        if predicate(qid, lab):
            found.append(qid)
            if len(found) >= n:
                break
    return found

def category_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def dump_category(title, qids, system, explanation):
    """Print full details for a list of qids in a given category."""
    category_header(title)
    print(f"  Total questions in this category: {len(qids)}")
    print()
    print(f"  ⟹ {explanation}")
    print()

    for i, qid in enumerate(qids[:SAMPLES_PER_CATEGORY], 1):
        lab = labels[qid]
        q = questions[qid]
        gold_answers = [lab["answer"]] + lab.get("aliases", [])
        pred = {"closed_book": closed, "dense": dense, "oracle": oracle}[system][qid]

        print_sample_separator(i)
        extra = [
            ("EM:", f"{em(pred['answer'], gold_answers)}"),
            ("F1:", f"{max_f1(pred['answer'], gold_answers):.3f}"),
        ]

        # Specific extra info for the system
        if system == "dense":
            missing_gold = [g for g in lab["gold_doc_ids"] if g not in pred["retrieved"][:10]]
            found_gold = [g for g in lab["gold_doc_ids"] if g in pred["retrieved"][:10]]
            extra.append(("gold in top-10:", f"{found_gold}"))
            extra.append(("gold MISSED:", f"{missing_gold}"))
            extra.append(("Recall@10:", f"{recall_at_k(pred['retrieved'], lab['gold_doc_ids'], 10):.3f}"))
            extra.append(("HitAll@10:", f"{hitall_at_k(pred['retrieved'], lab['gold_doc_ids'], 10):.3f}"))

        print_sample(qid, q["question"], pred["answer"], lab["answer"], lab["gold_doc_ids"], pred["retrieved"], extra)

        # Show the first gold passage snippet (helps understand context)
        print(f"  first gold passage:")
        print_passage_snippet(lab["gold_doc_ids"][0])
        print()


# ============ CATEGORY 1 ============
def cat1_filter(qid, lab):
    gold = [lab["answer"]] + lab.get("aliases", [])
    return em(closed[qid]["answer"], gold) == 0 and em(dense[qid]["answer"], gold) == 1

cat1_qids = find_samples(cat1_filter)
if cat1_qids:
    dump_category(
        "CATEGORY 1 — Closed-book WRONG, Dense RAG CORRECT (retrieval helped)",
        cat1_qids,
        system="dense",
        explanation="Demonstrates the value added by retrieval. The LLM alone failed, but with correct passages it answered correctly.",
    )

# ============ CATEGORY 2 ============
def cat2_filter(qid, lab):
    if questions[qid]["type"] != "bridge_comparison":
        return False
    gold = [lab["answer"]] + lab.get("aliases", [])
    if em(dense[qid]["answer"], gold) == 1:
        return False
    # Only pick cases where some golds are missing (i.e., real retrieval failure)
    missing = [g for g in lab["gold_doc_ids"] if g not in dense[qid]["retrieved"][:10]]
    return len(missing) >= 2  # at least 2 gold docs missed

cat2_qids = find_samples(cat2_filter)
if cat2_qids:
    dump_category(
        "CATEGORY 2 — Dense WRONG on bridge_comparison (retrieval missed gold docs)",
        cat2_qids,
        system="dense",
        explanation="THE CORE CHALLENGE. The retriever finds the main entity but misses intermediate entities (directors) whose names are not in the question. This is exactly what GraphRAG is designed to solve.",
    )

# ============ CATEGORY 3 ============
def cat3_filter(qid, lab):
    gold = [lab["answer"]] + lab.get("aliases", [])
    e = em(dense[qid]["answer"], gold)
    f = max_f1(dense[qid]["answer"], gold)
    return e == 0 and f > 0.3

cat3_qids = find_samples(cat3_filter)
if cat3_qids:
    dump_category(
        "CATEGORY 3 — Dense PARTIAL credit (F1 > 0.3, EM = 0)",
        cat3_qids,
        system="dense",
        explanation="The model found part of the answer or a related answer, but not the exact form. This is a near-miss that could be fixed with better retrieval or generation.",
    )

# ============ CATEGORY 4 ============
def cat4_filter(qid, lab):
    gold = [lab["answer"]] + lab.get("aliases", [])
    return em(oracle[qid]["answer"], gold) == 0

cat4_qids = find_samples(cat4_filter)
if cat4_qids:
    dump_category(
        "CATEGORY 4 — Oracle WRONG (gold context given, LLM still failed)",
        cat4_qids,
        system="oracle",
        explanation="Retrieval was perfect (all gold docs in context), but the LLM failed. This is a pure LLM/reasoning error — not a retrieval error.",
    )

# ============ CATEGORY 5 ============
def cat5_filter(qid, lab):
    gold = [lab["answer"]] + lab.get("aliases", [])
    if em(dense[qid]["answer"], gold) == 1:
        return False
    # All gold docs are retrieved
    all_found = all(g in dense[qid]["retrieved"][:10] for g in lab["gold_doc_ids"])
    return all_found

cat5_qids = find_samples(cat5_filter)
if cat5_qids:
    dump_category(
        "CATEGORY 5 — Dense retrieved ALL gold docs, but answer WRONG",
        cat5_qids,
        system="dense",
        explanation="Retrieval was perfect but the LLM failed to use the context. Likely causes: long context (lost-in-the-middle), too many distractor passages, or prompt issue.",
    )

# ============ CATEGORY 6 (bonus) ============
def cat6_filter(qid, lab):
    # Case where Dense MISSED gold, but Oracle got it right → isolates retrieval bottleneck
    gold = [lab["answer"]] + lab.get("aliases", [])
    dense_wrong = em(dense[qid]["answer"], gold) == 0
    oracle_right = em(oracle[qid]["answer"], gold) == 1
    missing_gold = [g for g in lab["gold_doc_ids"] if g not in dense[qid]["retrieved"][:10]]
    return dense_wrong and oracle_right and len(missing_gold) >= 1

cat6_qids = find_samples(cat6_filter)
if cat6_qids:
    dump_category(
        "CATEGORY 6 — Dense MISSED gold, but Oracle CORRECT (retrieval is the bottleneck)",
        cat6_qids,
        system="dense",
        explanation="Isolates the retrieval error: if gold had been retrieved, the LLM would have answered correctly (as Oracle proves). Fixing retrieval fixes the error.",
    )


# ============ SUMMARY ============
def print_summary():
    category_header("SUMMARY — Error Distribution")
    print()

    # Recompute all categories
    cat1 = [q for q, l in labels.items() if cat1_filter(q, l)]
    cat2 = [q for q, l in labels.items() if cat2_filter(q, l)]
    cat3 = [q for q, l in labels.items() if cat3_filter(q, l)]
    cat4 = [q for q, l in labels.items() if cat4_filter(q, l)]
    cat5 = [q for q, l in labels.items() if cat5_filter(q, l)]
    cat6 = [q for q, l in labels.items() if cat6_filter(q, l)]

    total_dense_wrong = sum(
        1 for q, l in labels.items()
        if em(dense[q]["answer"], [l["answer"]] + l.get("aliases", [])) == 0
    )
    total_oracle_wrong = sum(
        1 for q, l in labels.items()
        if em(oracle[q]["answer"], [l["answer"]] + l.get("aliases", [])) == 0
    )

    print(f"  Total Dense WRONG (EM=0):     {total_dense_wrong} / {len(labels)}")
    print(f"  Total Oracle WRONG (EM=0):    {total_oracle_wrong} / {len(labels)}")
    print()
    print(f"  Category 1 (Closed❌→Dense✅):                {len(cat1)} samples")
    print(f"  Category 2 (bridge_comparison, missing gold):  {len(cat2)} samples")
    print(f"  Category 3 (Dense F1>0.3 but EM=0):            {len(cat3)} samples")
    print(f"  Category 4 (Oracle ❌ despite gold context):   {len(cat4)} samples")
    print(f"  Category 5 (all gold retrieved but answer❌):  {len(cat5)} samples")
    print(f"  Category 6 (Dense missed gold → Oracle✅):     {len(cat6)} samples")
    print()
    print("  ── Interpretation ──")
    print()
    print(f"  Of {total_dense_wrong} Dense errors:")
    print(f"    • {len(cat2)} are on bridge_comparison with missed gold docs (main challenge)")
    print(f"    • {len(cat5)} happen even when all gold docs ARE retrieved (LLM limitation)")
    print(f"    • {len(cat6)} would be fixed if retrieval were perfect (retrieval bottleneck)")
    print(f"  Of {total_oracle_wrong} Oracle errors:")
    print(f"    • all are pure LLM reasoning failures (retrieval was perfect)")
    print()


# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 80)
    print("  ERROR ANALYSIS — Phase 1")
    print("=" * 80)
    print(f"  Loaded: {len(questions)} questions, {len(labels)} labels, {len(corpus)} passages")
    print()

    # Print all category content to console (also captured to file via tee)
    if not cat1_qids:
        print("[Category 1: no samples found]")
    if not cat2_qids:
        print("[Category 2: no samples found]")
    if not cat3_qids:
        print("[Category 3: no samples found]")
    if not cat4_qids:
        print("[Category 4: no samples found]")
    if not cat5_qids:
        print("[Category 5: no samples found]")
    if not cat6_qids:
        print("[Category 6: no samples found]")

    print_summary()

    print("=" * 80)
    print("  END")
    print("=" * 80)