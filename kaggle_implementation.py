# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# %%
# ==== Cell 1: Environment setup ====
!pip uninstall -y vllm torch torchaudio torchvision transformers
!pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
!pip install vllm==0.11.0
!pip install "transformers>=4.40.0,<5.0.0"
!pip install sentence-transformers accelerate

# %%
# ==== Cell 2: Verify installs ====
import torch, torchaudio, vllm, transformers
print(f"torch:        {torch.__version__}")
print(f"torchaudio:   {torchaudio.__version__}")
print(f"vllm:         {vllm.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# %%
# ==== Cell 3: Copy course pack to /kaggle/working ====
import shutil
from pathlib import Path

SRC = Path("/kaggle/input/datasets/foroughkoohi/graphrag-qa-data")
DST = Path("/kaggle/working")
for item in SRC.iterdir():
    dst = DST / item.name
    if item.is_dir():
        shutil.copytree(item, dst, dirs_exist_ok=True)
    else:
        shutil.copy(item, dst)

!ls -la /kaggle/working

# %%
# ==== Cell 4: Common imports and config ====
import os
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import json
import time
import yaml
from pathlib import Path

PACK = Path("/kaggle/working/data/course_pack_v1")
INDEX_DIR = Path("/kaggle/working/cache/index")
MODEL_PATH = "/kaggle/input/models/alphadua/qwen3-8b-awq-community/transformers/1/1"
K = 10  # <-- FIX: was 5

# Shared prompt: strict, identical for dev and hidden
PROMPT_TEMPLATE = """Answer the question with the SHORTEST possible answer.
CRITICAL RULES:
- If the answer is an entity, give ONLY the entity name
- If the answer is a date, give ONLY the date
- If the answer is a place, give ONLY the place name
- If it's a yes/no question, answer exactly 'yes' or 'no'
- DO NOT write sentences
- DO NOT explain
- DO NOT start with "based on" or "the context"
- Use lowercase
- No punctuation

{context_block}Question: {question}
Answer:"""

def write_config(out_dir, system_name, split, description, embedder, retrieval_unit, retrieval_k, command):
    cfg = {
        "system_name": system_name,
        "phase": 1,
        "split": split,
        "description": description,
        "models": {
            "llm": {
                "name": "alphadua/qwen3-8b-awq-community",
                "base_model": "Qwen/Qwen3-8B",
                "quantization": "awq",
                "where": "Kaggle T4, vLLM 0.11.0",
                "temperature": 0,
            },
            "embedder": embedder,
            "reranker": None,
        },
        "settings": {
            "retrieval": {"unit": retrieval_unit, "k": retrieval_k},
            "max_new_tokens": 30,
            "prompt_template": "strict short-answer",
        },
        "command": command,
        "date": time.strftime("%Y-%m-%d"),
    }
    with open(out_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, sort_keys=False)

print(f"Config ready. k = {K}")


# %%
# ==== Cell 5: Load LLM once and reuse ====
from vllm import LLM, SamplingParams

print(f"Loading LLM: {MODEL_PATH}")
llm = LLM(
    model=MODEL_PATH,
    quantization="awq",
    dtype="half",
    max_model_len=4096,
    gpu_memory_utilization=0.90,
    enforce_eager=True,
)
sampling = SamplingParams(temperature=0, max_tokens=30)
print("LLM ready")

# %%
# ==== Cell 6: Load embedder and index (used for dense) ====
import numpy as np
from sentence_transformers import SentenceTransformer

embeddings = np.load(INDEX_DIR / "embeddings.npy")
with open(INDEX_DIR / "doc_ids.json", encoding="utf-8") as f:
    doc_ids = json.load(f)
print(f"Index loaded: {embeddings.shape}")

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("Embedder ready")

corpus = {
    p["doc_id"]: p["text"]
    for p in (json.loads(l) for l in open(PACK / "corpus.jsonl", encoding="utf-8") if l.strip())
}


# %%
# ==== Helper functions ====
def clean_answer(text):
    return text.strip().split("\n")[0].strip().rstrip(".").lower()

def retrieve_top_k(questions, k=K):
    q_texts = [q["question"] for q in questions]
    q_embs = embedder.encode(q_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    scores = q_embs @ embeddings.T
    top_idx = np.argsort(-scores, axis=1)[:, :k]
    return [[doc_ids[i] for i in row] for row in top_idx]

def build_prompts(questions, retrieved_ids=None, gold_map=None):
    """Build prompts for closed_book (retrieved_ids=None, gold_map=None),
       dense (retrieved_ids provided), or oracle (gold_map provided)."""
    prompts = []
    for q in questions:
        if gold_map is not None:
            rids = gold_map[q["qid"]]["gold_doc_ids"]
        elif retrieved_ids is not None:
            rids = retrieved_ids[len(prompts)]
        else:
            rids = []
        if rids:
            ctx = "\n\n".join(f"- {corpus[did]}" for did in rids) + "\n\nContext:\n"
            ctx = "Context:\n" + "\n\n".join(f"- {corpus[did]}" for did in rids) + "\n\n"
        else:
            ctx = ""
        prompts.append(PROMPT_TEMPLATE.format(context_block=ctx, question=q["question"]))
    return prompts

def save_predictions(out_dir, questions, answers, retrieved_lists):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for q, a, rids in zip(questions, answers, retrieved_lists):
            f.write(json.dumps({
                "qid": q["qid"],
                "answer": a,
                "retrieved": rids,
            }, ensure_ascii=False) + "\n")
    return path

print("Helpers ready")

# %%
# ==== Cell 7: CLOSED-BOOK on dev ====
split = "dev"
questions = [json.loads(l) for l in open(PACK / f"questions_{split}.jsonl", encoding="utf-8") if l.strip()]
print(f"{split} questions: {len(questions)}")

prompts = build_prompts(questions)  # no context
t0 = time.time()
outputs = llm.generate(prompts, sampling)
print(f"closed-book {split}: {time.time()-t0:.1f}s")

answers = [clean_answer(o.outputs[0].text) for o in outputs]
out = Path(f"/kaggle/working/runs/{split}/closed_book")
path = save_predictions(out, questions, answers, [[] for _ in questions])
write_config(out, "closed_book", split,
             "Qwen3-8B AWQ via vLLM, no retrieval, strict prompt",
             embedder=None, retrieval_unit=None, retrieval_k=0,
             command=f"Kaggle notebook: closed-book on {split}")
print(f"Saved: {path}")

# %%
# ==== Cell 8: CLOSED-BOOK on hidden ====
split = "hidden"
questions = [json.loads(l) for l in open(PACK / f"questions_{split}.jsonl", encoding="utf-8") if l.strip()]
print(f"{split} questions: {len(questions)}")

prompts = build_prompts(questions)
t0 = time.time()
outputs = llm.generate(prompts, sampling)
print(f"closed-book {split}: {time.time()-t0:.1f}s")

answers = [clean_answer(o.outputs[0].text) for o in outputs]
out = Path(f"/kaggle/working/runs/{split}/closed_book")
path = save_predictions(out, questions, answers, [[] for _ in questions])
write_config(out, "closed_book", split,
             "Qwen3-8B AWQ via vLLM, no retrieval, strict prompt",
             embedder=None, retrieval_unit=None, retrieval_k=0,
             command=f"Kaggle notebook: closed-book on {split}")
print(f"Saved: {path}")

# %%
# ==== Cell 9: DENSE on dev (k=10) ====
split = "dev"
questions = [json.loads(l) for l in open(PACK / f"questions_{split}.jsonl", encoding="utf-8") if l.strip()]

retrieved_ids = retrieve_top_k(questions, k=K)
lengths = sorted(set(len(r) for r in retrieved_ids))
print(f"retrieved lengths: {lengths}")  # should be [10]
assert lengths == [K], f"BUG: lengths={lengths}, expected [{K}]"

prompts = build_prompts(questions, retrieved_ids=retrieved_ids)
t0 = time.time()
outputs = llm.generate(prompts, sampling)
print(f"dense {split}: {time.time()-t0:.1f}s")

answers = [clean_answer(o.outputs[0].text) for o in outputs]
out = Path(f"/kaggle/working/runs/{split}/dense")
path = save_predictions(out, questions, answers, retrieved_ids)
write_config(out, "dense", split,
             f"bge-small dense retrieval top-{K}, Qwen3-8B AWQ via vLLM, strict prompt",
             embedder="BAAI/bge-small-en-v1.5", retrieval_unit="passage", retrieval_k=K,
             command=f"Kaggle notebook: dense on {split}")
print(f"Saved: {path}")

# %%
# ==== Cell 10: DENSE on hidden (k=10) ====
split = "hidden"
questions = [json.loads(l) for l in open(PACK / f"questions_{split}.jsonl", encoding="utf-8") if l.strip()]

retrieved_ids = retrieve_top_k(questions, k=K)
lengths = sorted(set(len(r) for r in retrieved_ids))
print(f"retrieved lengths: {lengths}")  # should be [10]
assert lengths == [K], f"BUG: lengths={lengths}, expected [{K}]"

prompts = build_prompts(questions, retrieved_ids=retrieved_ids)
t0 = time.time()
outputs = llm.generate(prompts, sampling)
print(f"dense {split}: {time.time()-t0:.1f}s")

answers = [clean_answer(o.outputs[0].text) for o in outputs]
out = Path(f"/kaggle/working/runs/{split}/dense")
path = save_predictions(out, questions, answers, retrieved_ids)
write_config(out, "dense", split,
             f"bge-small dense retrieval top-{K}, Qwen3-8B AWQ via vLLM, strict prompt",
             embedder="BAAI/bge-small-en-v1.5", retrieval_unit="passage", retrieval_k=K,
             command=f"Kaggle notebook: dense on {split}")
print(f"Saved: {path}")


# %%
# ==== Cell 11: ORACLE on dev ====
split = "dev"
questions = [json.loads(l) for l in open(PACK / f"questions_{split}.jsonl", encoding="utf-8") if l.strip()]
labels = {l["qid"]: l for l in (json.loads(l) for l in open(PACK / f"labels_{split}.jsonl", encoding="utf-8") if l.strip())}

gold_lists = [labels[q["qid"]]["gold_doc_ids"] for q in questions]
lengths = sorted(set(len(g) for g in gold_lists))
print(f"oracle gold lengths: {lengths}")  # should be [1, 2, 4]

prompts = build_prompts(questions, gold_map=labels)
t0 = time.time()
outputs = llm.generate(prompts, sampling)
print(f"oracle {split}: {time.time()-t0:.1f}s")

answers = [clean_answer(o.outputs[0].text) for o in outputs]
out = Path(f"/kaggle/working/runs/{split}/oracle")
path = save_predictions(out, questions, answers, gold_lists)
write_config(out, "oracle", split,
             "gold_doc_ids as context, Qwen3-8B AWQ via vLLM, strict prompt",
             embedder=None, retrieval_unit="oracle", retrieval_k="all gold",
             command=f"Kaggle notebook: oracle on {split}")
print(f"Saved: {path}")


# %%
# ==== Cell 12: Free GPU memory before scoring ====
import gc, torch
del llm
gc.collect()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"GPU free: {free/1e9:.2f} GB / {total/1e9:.2f} GB")

# %%
# ==== Cell 13: Validate all prediction files ====
print("=" * 60)
print("VALIDATION")
print("=" * 60)
for split in ["dev", "hidden"]:
    for sys in ["closed_book", "dense", "oracle"]:
        p = Path(f"/kaggle/working/runs/{split}/{sys}/predictions.jsonl")
        if not p.exists():
            print(f"{split}/{sys}: NOT FOUND (ok if oracle on hidden)")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
        lengths = sorted(set(len(r["retrieved"]) for r in rows))
        cfg = Path(f"/kaggle/working/runs/{split}/{sys}/run_config.yaml")
        print(f"{split}/{sys}: {len(rows)} rows, retrieved lengths={lengths}, config={'OK' if cfg.exists() else 'MISSING'}")

# %%
# ==== Cell 14: Score dev systems ====
!python /kaggle/working/score.py /kaggle/working/runs/dev/closed_book/predictions.jsonl --pack /kaggle/working/data/course_pack_v1 --split dev

# %%
!python /kaggle/working/score.py /kaggle/working/runs/dev/dense/predictions.jsonl --pack /kaggle/working/data/course_pack_v1 --split dev

# %%
!python /kaggle/working/score.py /kaggle/working/runs/dev/oracle/predictions.jsonl --pack /kaggle/working/data/course_pack_v1 --split dev

# %%
# ==== Cell 15: Paired comparisons ====
!python /kaggle/working/score.py /kaggle/working/runs/dev/dense/predictions.jsonl --pack /kaggle/working/data/course_pack_v1 --split dev --compare /kaggle/working/runs/dev/closed_book/predictions.jsonl

# %%
!python /kaggle/working/score.py /kaggle/working/runs/dev/oracle/predictions.jsonl --pack /kaggle/working/data/course_pack_v1 --split dev --compare /kaggle/working/runs/dev/dense/predictions.jsonl


# %%
# ==== Cell 16: Zip everything and provide download link ====
import shutil
from IPython.display import FileLink

# Only zip what's needed (predictions + configs)
zpath = shutil.make_archive("/kaggle/working/runs_clean", "zip", "/kaggle/working/runs")
print(f"Created: {zpath}")
FileLink("runs_clean.zip")


