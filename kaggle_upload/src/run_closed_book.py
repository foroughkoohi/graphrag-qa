"""
Closed-book system: LLM answers without any retrieved context.
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
from tqdm import tqdm


MODEL_NAME = "google/gemma-2-2b-it"
PACK = PROJECT_ROOT / "data" / "course_pack_v1"


# ---------------- Prompt ----------------

PROMPT_TEMPLATE = """Answer the question with a SHORT answer.
Rules:
- Use lowercase only
- No punctuation at the end
- For yes/no questions, answer exactly 'yes' or 'no'
- Do not explain

Question: {question}
Answer:"""


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_model():
    print(f"Loading model: {MODEL_NAME}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f} seconds")
    return model, tokenizer


def generate_answer(model, tokenizer, question, max_new_tokens=50):
    prompt = PROMPT_TEMPLATE.format(question=question)
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    # Handle both old (Tensor) and new (BatchEncoding) transformers versions
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs
    else:
        input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    answer = response.strip().split("\n")[0].strip()

    answer = answer.rstrip(".")
    # lowercase
    answer = answer.lower()
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "hidden"])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only the first N questions (for a quick test)")
    args = ap.parse_args()

    # Output path
    out_dir = Path(args.out_dir) if args.out_dir else (
        PROJECT_ROOT / "runs" / args.split / "closed_book"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {PACK}")
    print(f"Output: {out_dir}")
    print(f"Split: {args.split}")

    # Load questions
    questions = load_jsonl(PACK / f"questions_{args.split}.jsonl")
    if args.limit:
        questions = questions[:args.limit]
    print(f"Number of questions: {len(questions)}")

    # Load model
    model, tokenizer = load_model()

    # Main loop
    print(f"\nStarting inference...")
    t_start = time.time()
    results = []

    for q in tqdm(questions, desc="Generating"):
        answer = generate_answer(model, tokenizer, q["question"])
        results.append({
            "qid": q["qid"],
            "answer": answer,
            "retrieved": [],  # closed-book: no retrieved context
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
        "system_name": "closed_book",
        "phase": 1,
        "split": args.split,
        "description": "LLM (gemma-2-2b-it) on CPU, no retrieval, greedy decoding",
        "models": {
            "llm": {
                "name": MODEL_NAME,
                "where": "Local CPU (AMD Ryzen 7 3700U)",
                "temperature": 0,
            },
            "embedder": None,
            "reranker": None,
        },
        "settings": {
            "retrieval": {"unit": None, "k": 0},
            "max_new_tokens": 50,
            "prompt_template": "strict short-answer",
        },
        "command": f"python src/run_closed_book.py --split {args.split}",
        "date": time.strftime("%Y-%m-%d"),
    }

    config_path = out_dir / "run_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(run_config, f, allow_unicode=True, sort_keys=False)
    print(f"run_config: {config_path}")


if __name__ == "__main__":
    main()