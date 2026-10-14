"""
Oracle-context system: gives gold_doc_ids directly to the LLM.
This is the performance ceiling.
"""
import argparse
import json
import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "hub"))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from tqdm import tqdm


PACK = PROJECT_ROOT / "data" / "course_pack_v1"
LLM_MODEL_NAME = "google/gemma-2-2b-it"

PROMPT_TEMPLATE = """Answer the question based on the context below.
Rules:
- Use lowercase only
- No punctuation at the end
- For yes/no questions, answer exactly 'yes' or 'no'
- Do not explain
- Just give the answer, nothing else

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


def pick_dtype_and_device():
    
    if torch.cuda.is_available():
        return torch.float16, "auto"
    
    try:
        if torch.backends.cpu.get_cpu_capability() == "AVX512":
            return torch.bfloat16, "cpu"
    except Exception:
        pass
    return torch.float32, "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # oracle only makes sense on dev (because we have labels)
    out_dir = PROJECT_ROOT / "runs" / args.split / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")

    # Load
    questions = load_jsonl(PACK / f"questions_{args.split}.jsonl")
    labels = {l["qid"]: l for l in load_jsonl(PACK / f"labels_{args.split}.jsonl")}
    corpus = {p["doc_id"]: p["text"] for p in load_jsonl(PACK / "corpus.jsonl")}

    if args.limit:
        questions = questions[:args.limit]
    print(f"Number of questions: {len(questions)}")

    # Load LLM
    print(f"Loading LLM: {LLM_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    dtype, device_map = pick_dtype_and_device()
    llm = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    llm.eval()
    model_device = next(llm.parameters()).device

    # Loop
    print("\nStarting...")
    t_start = time.time()
    results = []

    for q in tqdm(questions, desc="Oracle"):
        gold_ids = labels[q["qid"]]["gold_doc_ids"]
        context_texts = [corpus[did] for did in gold_ids]
        context_text = "\n\n".join(f"- {c}" for c in context_texts)

        prompt = PROMPT_TEMPLATE.format(context=context_text, question=q["question"])
        messages = [{"role": "user", "content": prompt}]

        
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=False,
        )

        
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
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = response.strip().split("\n")[0].strip().rstrip(".").lower()

        results.append({
            "qid": q["qid"],
            "answer": answer,
            "retrieved": gold_ids,  # oracle: we put the gold ids
        })

    total_time = time.time() - t_start
    print(f"\n{len(results)} questions in {total_time:.1f} seconds")

    # Save
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    run_config = {
        "system_name": "oracle_context",
        "phase": 1,
        "split": args.split,
        "description": "gold_doc_ids as context, gemma-2-2b-it on CPU",
        "models": {
            "llm": {"name": LLM_MODEL_NAME, "where": "Local CPU", "temperature": 0},
            "embedder": None,
            "reranker": None,
        },
        "settings": {
            "retrieval": {"unit": "oracle", "k": "all gold"},
            "max_new_tokens": 30,
        },
        "command": f"python src/run_oracle.py --split {args.split}",
        "date": time.strftime("%Y-%m-%d"),
    }

    with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)

    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()