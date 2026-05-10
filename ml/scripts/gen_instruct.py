"""
gen_instruct.py
This script will generate instruction-following examples from the cleaned data.

Supports resuming — if the script crashes, re-running it will skip
already-processed examples and continue from where it left off.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
openrouter_token = os.getenv("OPENROUTER_TOKEN")
base_model = os.getenv("BASE_MODEL")

# Config
MAX_SAMPLES = 10_000  # Target number of instruction examples to generate
BASE_DELAY = 1.0  # Base seconds between API calls
MAX_RETRIES = 3  # Maximum retries per failed request
OUTPUT_FILE = "../data/instruct/zig_instruct_data.jsonl"
INPUT_FILE = "../data/cleaned/zig_cleaned_data.jsonl"
MODEL = base_model
API_BASE_URL = "https://openrouter.ai/api/v1"

if not openrouter_token:
    raise ValueError("OPENROUTER_TOKEN not found in environment variables")

if not MODEL:
    raise ValueError("BASE_MODEL not found in environment variables")


def load_already_processed(output_file):
    """Load hashes of code already written to the output file for resume support."""
    seen = set()
    if not os.path.exists(output_file):
        return seen
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                seen.add(record.get("code", ""))
            except json.JSONDecodeError:
                continue
    print(f"Resuming — skipping {len(seen)} already processed examples.")
    return seen


def call_openrouter(code, max_retries=MAX_RETRIES):
    """Generate instruction for a Zig code snippet with retry + exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {openrouter_token}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/your-repo",
                    "X-Title": "Zig Model Instruction Generator",
                },
                json={
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a coding instructor. Given a Zig source code snippet, "
                                "write a single clear instruction sentence that describes what the "
                                "code does, as if you are asking someone to write it from scratch. "
                                "Return only the instruction sentence, without any additional text."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"Code:\n```zig\n{code}\n```",
                        },
                    ],
                },
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            return result
        except requests.exceptions.RequestException as e:
            wait_time = BASE_DELAY * (2**attempt)  # Exponential backoff: 1s, 2s, 4s
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts, skipping.")
                return None
    return None


def generate_instruction():
    """Generate instruction-following dataset from cleaned Zig code."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load already-processed examples to support resuming
    already_processed = load_already_processed(OUTPUT_FILE)
    already_done = len(already_processed)

    total_skipped = 0
    total_written = 0
    total_errors = 0

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(OUTPUT_FILE, "a", encoding="utf-8") as outfile,
    ):
        for line in infile:
            # Stop once we hit the target
            if already_done + total_written >= MAX_SAMPLES:
                print(f"Reached MAX_SAMPLES limit of {MAX_SAMPLES}.")
                break

            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                total_errors += 1
                continue

            code = data.get("content", "")
            if not code:
                continue

            # Skip if already processed (resume support)
            if code in already_processed:
                total_skipped += 1
                continue

            # Generate instruction
            instruction = call_openrouter(code)
            if instruction is None:
                total_errors += 1
                continue

            # Write instruction-following example
            example = {
                "instruction": instruction,
                "code": code,
                "repo_name": data.get("repo_name", ""),
                "path": data.get("path", ""),
                "language": data.get("language", "Zig"),
            }
            outfile.write(json.dumps(example) + "\n")
            outfile.flush()  # Flush after every write so progress isn't lost on crash
            total_written += 1

            if total_written % 10 == 0:
                total_so_far = already_done + total_written
                print(f"Progress: {total_so_far}/{MAX_SAMPLES} examples written...")

            # Rate limiter
            time.sleep(BASE_DELAY)

    print("\n=== Generation Complete ===")
    print(f"Target samples: {MAX_SAMPLES}")
    print(f"Already done (resumed): {already_done}")
    print(f"Newly written: {total_written}")
    print(f"Total in output: {already_done + total_written}")
    print(f"Errors: {total_errors}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_instruction()
