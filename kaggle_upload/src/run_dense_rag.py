"""
Dense RAG system:
1. Uses bge-small to find relevant texts
2. Gives the texts + question to the LLM
3. Gets the answer
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from tqdm import tqdm


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
INDEX_DIR = PROJECT_ROOT / "cache" / "index"

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LLM_MODEL_NAME = "google/gemma-2-2b-it"


# ---------------- RAG prompt ----------------

PROMPT_TEMPLATE = """Answer the question based on the context below.
Rules:
- Use lowercase only
- No punctuation at the end
- For yes/no questions, answer exactly 'yes' or 'no'
- Do not explain
- Just give the answer, nothing else
- If the answer is not in the context, give your best guess

Context:
{context}

Question: {question}
Answer:"""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_index():
    embeddings = np.load(INDEX_DIR / "embeddings.npy")
    with open(INDEX_DIR / "doc_ids.json", encoding="utf-8") as f:
        doc_ids = json.load(f)
    return embeddings, doc_ids


def load_models():
    print(f"Loading embedder: {EMBED_MODEL_NAME}")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Loading LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    
    if torch.cuda.is_available():
        dtype = torch.float16
        device_map = "auto"
    else:
        
        try:
            if torch.backends.cpu.get_cpu_capability() == "AVX512":
                dtype = torch.bfloat16
            else:
                dtype = torch.float32
        except Exception:
            dtype = torch.float32
        device_map = "cpu"

    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    llm.eval()
    return embedder, llm, tokenizer


def retrieve(question, embedder, embeddings, doc_ids, k=5):
    q_emb = embedder.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    scores = embeddings @ q_emb
    top_idx = np.argsort(-scores)[:k]
    return [doc_ids[i] for i in top_idx]


def generate_answer(llm, tokenizer, question, contexts, max_new_tokens=30):
    context_text = "\n\n".join(f"- {c}" for c in contexts)
    prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
    messages = [{"role": "user", "content": prompt}]

    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=False,
    )

    
    model_device = next(llm.parameters()).device
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.to(model_device)
        input_len = inputs.shape[1]
        gen_kwargs = {"input_ids": inputs}
    else:
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        gen_kwargs = inputs

    with torch.no_grad():
        outputs = llm.generate(
            **gen_kwargs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    answer = response.strip().split("\n")[0].strip().rstrip(".").lower()
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "hidden"])
    ap.add_argument("--k", type=int, default=5, help="number of retrieved texts")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "runs" / args.split / "dense"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")
    print(f"split: {args.split}, k={args.k}")

    # Load questions
    questions = load_jsonl(PACK / f"questions_{args.split}.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    print(f"Number of questions: {len(questions)}")

    # Load corpus for context
    corpus = {p["doc_id"]: p["text"] for p in load_jsonl(PACK / "corpus.jsonl")}

    # Load index
    print("Loading index...")
    embeddings, doc_ids = load_index()
    print(f"Index loaded: {embeddings.shape}")

    # Load models
    embedder, llm, tokenizer = load_models()

    # Main loop
    print(f"\nStarting...")
    t_start = time.time()
    results = []

    for q in tqdm(questions, desc="Dense RAG"):
        # retrieval
        retrieved_ids = retrieve(q["question"], embedder, embeddings, doc_ids, k=args.k)
        # texts
        context_texts = [corpus[did] for did in retrieved_ids]
        # LLM
        answer = generate_answer(llm, tokenizer, q["question"], context_texts)
        results.append({
            "qid": q["qid"],
            "answer": answer,
            "retrieved": retrieved_ids,
        })

    total_time = time.time() - t_start
    print(f"\n{len(results)} questions in {total_time:.1f} seconds")
    print(f"   Average: {total_time / len(results):.1f} seconds/question")

    # Save predictions
    pred_path = out_dir / "predictions.jsonl"
    with open(pred_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"predictions: {pred_path}")

    # Save run_config
    run_config = {
        "system_name": "dense_rag",
        "phase": 1,
        "split": args.split,
        "description": f"bge-small dense retrieval top-{args.k}, gemma-2-2b-it on CPU",
        "models": {
            "llm": {
                "name": LLM_MODEL_NAME,
                "where": "Local CPU (AMD Ryzen 7 3700U)",
                "temperature": 0,
            },
            "embedder": EMBED_MODEL_NAME,
            "reranker": None,
        },
        "settings": {
            "retrieval": {"unit": "passage", "k": args.k},
            "max_new_tokens": 30,
            "prompt_template": "rag short-answer",
        },
        "command": f"python src/run_dense_rag.py --split {args.split} --k {args.k}",
        "date": time.strftime("%Y-%m-%d"),
    }

    config_path = out_dir / "run_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)
    print(f"run_config: {config_path}")


if __name__ == "__main__":
    main()