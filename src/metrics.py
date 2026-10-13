"""
Implementation of retrieval and answer evaluation metrics.
Based on Phase 1 Handout — without using external libraries.
"""
import math
import re
import string
from collections import Counter


# ============================================================
# Text Normalization
# ============================================================

def normalize(text):
    """
    Normalize text for comparison:
    - lowercase
    - remove punctuation
    - remove a/an/the
    - remove extra whitespace
    """
    if not text:
        return ""
    # lowercase
    text = text.lower()
    # remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    # remove a/an/the
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # remove extra whitespace
    text = " ".join(text.split())
    return text


# ============================================================
# Retrieval Metrics
# ============================================================

def recall_at_k(retrieved, gold, k):
    """
    Recall@k = |gold ∩ retrieved[:k]| / |gold|
    """
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & set(gold)) / len(gold)


def hitall_at_k(retrieved, gold, k):
    """
    HitAll@k = 1 if all gold docs are in the top-k retrieved, otherwise 0
    """
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return 1.0 if set(gold) <= top_k else 0.0


def mrr(retrieved, gold):
    """
    MRR = 1 / rank of the first gold doc in the retrieved list
    """
    gold_set = set(gold)
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in gold_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved, gold, k=10):
    """
    nDCG@k = DCG@k / IDCG@k
    """
    gold_set = set(gold)
    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], 1):
        if doc_id in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    # IDCG@k
    idcg = 0.0
    for i in range(1, min(len(gold), k) + 1):
        idcg += 1.0 / math.log2(i + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ============================================================
# Answer Metrics
# ============================================================

def exact_match(pred, gold_list):
    """
    EM = 1 if pred matches any gold answer or alias (after normalization)
    """
    if not pred:
        return 0.0
    pred_norm = normalize(pred)
    for gold in gold_list:
        if pred_norm == normalize(gold):
            return 1.0
    return 0.0


def f1_score(pred, gold):
    """
    Word-level F1 score between pred and gold
    """
    pred_norm = normalize(pred)
    gold_norm = normalize(gold)

    # Special case for yes/no/noanswer
    special = {"yes", "no", "noanswer"}
    if (pred_norm in special or gold_norm in special) and pred_norm != gold_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_f1(pred, gold_list):
    """
    Maximum F1 score between pred and all gold answers/aliases
    """
    if not pred:
        return 0.0
    return max(f1_score(pred, g) for g in gold_list)


# ============================================================
# Calculate all metrics for a single question
# ============================================================

def evaluate_one(prediction, label):
    """
    prediction: Dictionary with keys qid, answer, retrieved, (optional) cited, abstained
    label:      Dictionary from labels_dev.jsonl
    Returns:    Dictionary of metrics
    """
    retrieved = prediction.get("retrieved", [])
    gold_docs = label["gold_doc_ids"]

    # If abstained, consider the answer as empty
    answer = "" if prediction.get("abstained") else prediction.get("answer", "")

    gold_answers = [label["answer"]] + label.get("aliases", [])

    metrics = {}

    # ---- Retrieval ----
    for k in (2, 5, 10):
        metrics[f"recall@{k}"] = recall_at_k(retrieved, gold_docs, k)
        metrics[f"hitall@{k}"] = hitall_at_k(retrieved, gold_docs, k)

    metrics["mrr"] = mrr(retrieved, gold_docs)
    metrics["ndcg@10"] = ndcg_at_k(retrieved, gold_docs, k=10)

    # ---- Answer ----
    metrics["em"] = exact_match(answer, gold_answers)
    metrics["f1"] = max_f1(answer, gold_answers)

    # ---- Optional ----
    if "cited" in prediction:
        cited = set(prediction["cited"])
        gold_set = set(gold_docs)
        metrics["cite_precision"] = len(cited & gold_set) / len(cited) if cited else 0.0
        metrics["cite_recall"] = len(cited & gold_set) / len(gold_set)

    if "abstained" in prediction:
        metrics["abstained"] = 1.0 if prediction["abstained"] else 0.0

    return metrics