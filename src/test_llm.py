"""
Quick test of the language model on CPU
Model: google/gemma-2-2b-it
"""
import os
import time
from pathlib import Path

# Cache path inside the project drive
PROJECT_ROOT = Path(__file__).parent.parent
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "hub")
)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "google/gemma-2-2b-it"


def get_input_ids(tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    
    if isinstance(inputs, torch.Tensor):
        return inputs
    return inputs["input_ids"]


def main():
    print(f"Cache path: {os.environ['HF_HUB_CACHE']}")
    print(f"Loading model: {MODEL_NAME}")
    print("   (If this is the first time, the model will be downloaded - it may take a few minutes)")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # CPU only supports float32
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()

    print(f"Model loaded in {time.time() - t0:.1f} seconds")

    # ---- Test 1: General question ----
    prompt1 = "What is the capital of France? Answer in one word."
    print(f"\n{'='*60}")
    print(f"Prompt 1: {prompt1}")
    print("Generating answer...")

    messages = [{"role": "user", "content": prompt1}]
    input_ids = get_input_ids(tokenizer, messages)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Answer: {response!r}")
    print(f"Time: {gen_time:.1f} seconds")

    # ---- Test 2: Question from the dataset ----
    prompt2 = (
        "Which film has the director who was born later, "
        "The Third Side Of The River or Escape From L.A.?"
    )
    print(f"\n{'='*60}")
    print(f"Prompt 2 (from dataset): {prompt2}")
    print("Generating answer...")

    messages = [{"role": "user", "content": prompt2}]
    input_ids = get_input_ids(tokenizer, messages)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_time = time.time() - t0

    response = tokenizer.decode(
        outputs[0][input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    print(f"Answer: {response!r}")
    print(f"Time: {gen_time:.1f} seconds")

    print(f"\n{'='*60}")
    print("Test finished!")


if __name__ == "__main__":
    main()