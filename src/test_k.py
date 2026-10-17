import json

for split in ["dev", "hidden"]:
    for sys in ["closed_book", "dense", "oracle"]:
        path = f"runs/{split}/{sys}/predictions.jsonl"
        try:
            lengths = set()
            for line in open(path, encoding="utf-8"):
                if line.strip():
                    p = json.loads(line)
                    lengths.add(len(p["retrieved"]))
            print(f"{split:6s}/{sys:12s}: lengths = {sorted(lengths)}")
        except FileNotFoundError:
            print(f"{split:6s}/{sys:12s}: NOT FOUND")