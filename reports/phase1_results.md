# Phase 1 - Results

## Table 1: Overall comparison of the three systems (dev, n=300)

| System | EM | F1 | R@2 | R@5 | R@10 | HitAll@5 | MRR | nDCG@10 |
|---|---|---|---|---|---|---|---|---|
| closed-book | 18.0 | 21.3 | 0 | 0 | 0 | 0 | 0 | 0 |
| dense RAG | 23.7 | 33.3 | 62.7 | 71.5 | 71.5 | 43.0 | 97.6 | 75.8 |
| oracle (n=10) | 70.0 | 75.0 | 95.0 | 100 | 100 | 100 | 100 | 100 |

## Table 2: Dense RAG by question type

| Type | n | EM | F1 | R@5 | HitAll@5 | MRR |
|---|---|---|---|---|---|---|
| comparison | 73 | 38.4 | 47.4 | 99.3 | 98.6 | 99.3 |
| bridge_comparison | 66 | 27.3 | 33.6 | 52.3 | **1.5** | 97.7 |
| compositional | 125 | 19.2 | 24.8 | 65.2 | 31.2 | 98.4 |
| inference | 36 | **2.8** | 33.5 | 72.2 | 47.2 | 90.8 |

## Key findings

1. **RAG is more effective than closed-book**: EM +5.7, F1 +12
2. **Dense retrieval is excellent at simple retrieval**: MRR=97.6%
3. **HitAll@5 for bridge_comparison is a disaster**: 1.5%
4. **Small LLM has limitations**: Oracle 70% vs Dense 23.7%
5. **Multi-step reasoning (inference) is the hardest**: EM=2.8%

## Conclusion

- Metrics implementation complete (matching score.py)
- Three baseline systems (closed-book, dense, oracle)
- Prompt needs improvement for short answers
- GraphRAG needed for bridge_comparison
- Ready for Phase 2 (Hybrid RAG)