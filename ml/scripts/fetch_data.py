"""
fetch_data.py
This python file will pull The Stack v2/v1 from Huggingface

Specifically zig data
"""
import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login

load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

login(token=hf_token)

# Configs
OUTPUT_DIR = "../data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "zig_raw_data.jsonl")
MAX_SAMPLES = 50_000

# Setup
os.makedirs(OUTPUT_DIR, exist_ok=True) # this will only make the folder if not yet existing

# Load
ds = load_dataset(
    "bigcode/the-stack",
    data_dir="data/zig",
    split="train",
    streaming=True,
)

# Try saving to file

count = 0
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in ds:
        if count >= MAX_SAMPLES:
            break

        record = {
            "content": sample["content"],
            "repo_name": sample.get("repository_name", ""),
            "path": sample.get("path", ""),
            "language": "Zig",
        }

        f.write(json.dumps(record) + "\n")
        count += 1

print(f"Saved {count} samples to {OUTPUT_FILE}")