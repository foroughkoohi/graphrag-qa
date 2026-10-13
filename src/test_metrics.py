"""
Test metrics with manual examples.
"""
from metrics import (
    recall_at_k, hitall_at_k, mrr, ndcg_at_k,
    exact_match, f1_score, normalize
)


def test_normalize():
    print("--- Testing normalize ---")
    assert normalize("The Cat") == "cat"
    assert normalize("A Dog!") == "dog"
    assert normalize("  Hello   World  ") == "hello world"
    assert normalize("") == ""
    assert normalize("YES") == "yes"
    print("normalize works correctly\n")


def test_recall():
    print("--- Testing Recall@k ---")
    retrieved = ["p1", "p5", "p2", "p6", "p7", "p3", "p4"]
    gold = ["p1", "p2", "p3", "p4"]

    assert recall_at_k(retrieved, gold, 2) == 1/4  # p1 only
    assert recall_at_k(retrieved, gold, 5) == 2/4  # p1, p2
    assert recall_at_k(retrieved, gold, 10) == 4/4  # all
    print("Recall@k works correctly\n")


def test_hitall():
    print("--- Testing HitAll@k ---")
    retrieved = ["p1", "p5", "p2", "p6", "p7", "p3", "p4"]
    gold = ["p1", "p2", "p3", "p4"]

    assert hitall_at_k(retrieved, gold, 2) == 0.0  # p3, p4 are missing
    assert hitall_at_k(retrieved, gold, 5) == 0.0  # p3, p4 are missing
    assert hitall_at_k(retrieved, gold, 10) == 1.0  # all present
    print("HitAll@k works correctly\n")


def test_mrr():
    print("--- Testing MRR ---")
    retrieved = ["p5", "p6", "p1", "p2"]
    gold = ["p1", "p2", "p3", "p4"]
    # First gold = p1 at rank 3
    assert mrr(retrieved, gold) == 1/3

    # If gold is not in retrieved
    assert mrr(["p5", "p6"], gold) == 0.0
    print("MRR works correctly\n")


def test_ndcg():
    print("--- Testing nDCG@10 ---")
    # Simple example: all 4 gold docs are in ranks 1 to 4
    retrieved = ["p1", "p2", "p3", "p4"]
    gold = ["p1", "p2", "p3", "p4"]
    # nDCG should be 1.0 because all are in the best possible ranks
    ndcg = ndcg_at_k(retrieved, gold, 10)
    assert abs(ndcg - 1.0) < 1e-6, f"ndcg = {ndcg}"

    # Another example: gold docs at ranks 3 and 5
    retrieved = ["x", "y", "p1", "z", "p2"]
    gold = ["p1", "p2"]
    ndcg = ndcg_at_k(retrieved, gold, 10)
    print(f"  nDCG@10 with ranks 3 and 5: {ndcg:.4f}")
    print("nDCG@10 works correctly\n")


def test_em():
    print("--- Testing EM ---")
    assert exact_match("The Cat", ["the cat"]) == 1.0
    assert exact_match("cat", ["the cat"]) == 1.0  # after normalization
    assert exact_match("dog", ["cat"]) == 0.0
    assert exact_match("", ["cat"]) == 0.0
    # aliases
    assert exact_match("Third Side of the River", ["The Third Side of the River"]) == 1.0
    print("EM works correctly\n")


def test_f1():
    print("--- Testing F1 ---")
    # yes/no exception
    assert f1_score("yes", "no") == 0.0
    assert f1_score("yes", "yes") == 1.0
    # Normal F1
    f1 = f1_score("John Carpenter", "John Carpenter")
    assert f1 == 1.0
    f1 = f1_score("John", "John Carpenter")
    print(f"  F1('John', 'John Carpenter') = {f1:.4f}")
    assert 0 < f1 < 1
    print("F1 works correctly\n")


if __name__ == "__main__":
    test_normalize()
    test_recall()
    test_hitall()
    test_mrr()
    test_ndcg()
    test_em()
    test_f1()
    print("=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)