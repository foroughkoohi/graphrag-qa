"""
Exact Dense Retriever with Document Mask capability

This class is an "exact" dense retriever:
- Instead of FAISS or ANN, it uses brute-force (dot product of all vectors)
- It supports masking: you can remove certain documents from the results
- The mask is applied **before** top-k selection (per the handout requirement)
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_dir, model_name="BAAI/bge-small-en-v1.5"):
        """
        Args:
            index_dir: the folder containing embeddings.npy and doc_ids.json
            model_name: the embedding model name
        """
        index_dir = Path(index_dir)
        self.embeddings = np.load(index_dir / "embeddings.npy")
        with open(index_dir / "doc_ids.json", encoding="utf-8") as f:
            self.doc_ids = json.load(f)

        # Dictionary for fast O(1) lookup instead of list.index()
        self.doc_id_to_idx = {did: i for i, did in enumerate(self.doc_ids)}

        print(f"Retriever loaded: {len(self.doc_ids)} documents, {self.embeddings.shape[1]} dims")
        self.embedder = SentenceTransformer(model_name)

    def retrieve(self, question, k=5, exclude_ids=None):
        """
        Exact retrieval (brute force) with optional mask.

        Args:
            question: the question text
            k: number of returned results
            exclude_ids: list of doc_ids that should be removed from the results (mask)

        Returns:
            list of doc_ids in ranked order (best first)
        """
        # Question embedding
        q_emb = self.embedder.encode(
            question,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Exact similarity computation: dot product (since we normalized = cosine)
        scores = self.embeddings @ q_emb

        # ===== Apply mask before top-k selection =====
        # Per the handout requirement: the mask must be applied before top-k
        # otherwise the number of returned results might be less than k
        if exclude_ids:
            for eid in exclude_ids:
                if eid in self.doc_id_to_idx:
                    scores[self.doc_id_to_idx[eid]] = -np.inf

        # Select top-k
        top_idx = np.argsort(-scores)[:k]
        return [self.doc_ids[i] for i in top_idx]


if __name__ == "__main__":
    # Quick test
    import os
    PROJECT_ROOT = Path(__file__).parent.parent
    INDEX_DIR = PROJECT_ROOT / "cache" / "index"

    retriever = DenseRetriever(INDEX_DIR)

    q = "Which film has the director who was born later, The Third Side Of The River or Escape From L.A.?"
    print(f"\nQuestion: {q}")

    print("\n--- without mask ---")
    print(retriever.retrieve(q, k=5))

    print("\n--- with mask (removing the first 2 documents) ---")
    print(retriever.retrieve(q, k=5, exclude_ids=["p03489", "p01066"]))