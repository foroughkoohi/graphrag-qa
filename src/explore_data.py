"""
Data exploration script for course_pack_v1
"""
import json
from pathlib import Path
from collections import Counter

# Data pack path
PACK = Path("data/course_pack_v1")


def load_jsonl(path):
    """Read a jsonl file and return a list of dictionaries."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def explore_corpus():
    """Explore the text corpus."""
    print("=" * 60)
    print("Exploring corpus (corpus.jsonl)")
    print("=" * 60)

    corpus = load_jsonl(PACK / "corpus.jsonl")
    print(f"Total number of texts: {len(corpus)}")
    print(f"Keys in each text: {list(corpus[0].keys())}")

    # First sample
    print("\n--- First Sample ---")
    sample = corpus[0]
    print(f"doc_id: {sample['doc_id']}")
    print(f"title: {sample['title']}")
    print(f"text (first 100 chars): {sample['text'][:100]}...")
    print(f"Number of sentences: {len(sample['sentences'])}")
    print(f"First sentence: {sample['sentences'][0]}")

    # Text length statistics
    lengths = [len(p["text"].split()) for p in corpus]
    print(f"\n--- Text length statistics (by word) ---")
    print(f"Average: {sum(lengths) / len(lengths):.1f}")
    print(f"Shortest: {min(lengths)}")
    print(f"Longest: {max(lengths)}")
    print(f"Median: {sorted(lengths)[len(lengths) // 2]}")

    # Sentence count statistics
    sent_counts = [len(p["sentences"]) for p in corpus]
    print(f"\n--- Sentence count statistics ---")
    print(f"Average: {sum(sent_counts) / len(sent_counts):.1f}")
    print(f"Maximum: {max(sent_counts)}")

    return corpus


def explore_questions(split="dev"):
    """Explore the questions."""
    print("\n" + "=" * 60)
    print(f"Exploring questions (questions_{split}.jsonl)")
    print("=" * 60)

    questions = load_jsonl(PACK / f"questions_{split}.jsonl")
    print(f"Total number of questions: {len(questions)}")
    print(f"Keys in each question: {list(questions[0].keys())}")

    # Question type distribution
    types = Counter(q["type"] for q in questions)
    print(f"\n--- Question type distribution ---")
    for t, c in types.most_common():
        print(f"  {t}: {c} ({100 * c / len(questions):.1f}%)")

    # Question samples
    print("\n--- Question samples (one of each type) ---")
    seen_types = set()
    for q in questions:
        if q["type"] not in seen_types:
            seen_types.add(q["type"])
            print(f"\n[{q['type']}] {q['qid']}")
            print(f"  {q['question']}")

    return questions


def explore_labels():
    """Explore the gold labels."""
    print("\n" + "=" * 60)
    print("Exploring labels (labels_dev.jsonl)")
    print("=" * 60)

    labels = load_jsonl(PACK / "labels_dev.jsonl")
    print(f"Total number of labels: {len(labels)}")
    print(f"Keys in each label: {list(labels[0].keys())}")

    # First sample
    print("\n--- First Sample ---")
    sample = labels[0]
    for key, val in sample.items():
        if isinstance(val, list) and len(val) > 3:
            print(f"{key}: {val[:3]} ... (count: {len(val)})")
        else:
            print(f"{key}: {val}")

    # Distribution of gold_doc_ids count
    gold_counts = Counter(len(lab["gold_doc_ids"]) for lab in labels)
    print(f"\n--- Distribution of gold_doc_ids count per question ---")
    for n, c in sorted(gold_counts.items()):
        print(f"  {n} gold texts: {c} questions")

    # Short answer samples
    print(f"\n--- Short answer samples ---")
    for lab in labels[:5]:
        print(f"  {lab['qid']}: {lab['answer']!r} (type: {lab['type']})")

    return labels


def cross_check():
    """Check consistency between files."""
    print("\n" + "=" * 60)
    print("Checking consistency between files")
    print("=" * 60)

    corpus = load_jsonl(PACK / "corpus.jsonl")
    questions = load_jsonl(PACK / "questions_dev.jsonl")
    labels = load_jsonl(PACK / "labels_dev.jsonl")

    corpus_ids = {d["doc_id"] for d in corpus}
    question_ids = {q["qid"] for q in questions}
    label_ids = {lab["qid"] for lab in labels}

    print(f"Number of unique doc_ids in corpus: {len(corpus_ids)}")
    print(f"Number of unique qids in questions_dev: {len(question_ids)}")
    print(f"Number of unique qids in labels_dev: {len(label_ids)}")

    # Are qids identical?
    print(f"\nAre questions qids == labels qids? {question_ids == label_ids}")

    # Are gold_doc_ids present in corpus?
    missing = []
    for lab in labels:
        for doc_id in lab["gold_doc_ids"]:
            if doc_id not in corpus_ids:
                missing.append((lab["qid"], doc_id))

    if missing:
        print(f"\nWarning: {len(missing)} gold_doc_ids are not in the corpus!")
        for qid, doc_id in missing[:5]:
            print(f"  {qid}: {doc_id}")
    else:
        print(f"\nSuccess: All gold_doc_ids are present in the corpus.")


if __name__ == "__main__":
    explore_corpus()
    explore_questions("dev")
    explore_labels()
    cross_check()

    print("\n" + "=" * 60)
    print("Exploration completed successfully!")
    print("=" * 60)