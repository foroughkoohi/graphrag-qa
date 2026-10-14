Course pack v1 -- see the handout (data section) for the full description.

corpus.jsonl            3,882 passages: {"doc_id", "title", "text", "sentences": [...]}
questions_dev.jsonl     300 questions:  {"qid", "question", "type"}
labels_dev.jsonl        their labels:   {"qid", "answer", "aliases", "gold_doc_ids", "gold_sentences": [[doc_id, sentence_index], ...],
                                         "evidence_triples": [[subject, relation, object], ...], "type"}
questions_hidden.jsonl  300 questions without labels (submit predictions; the TAs score them)
kg_subset_passages.txt  200 doc ids for the Phase 3 extraction exercise

Question types: comparison, bridge_comparison, compositional, inference.
Score a predictions file:  python score.py <predictions.jsonl> --pack course_pack_v1 --split dev
