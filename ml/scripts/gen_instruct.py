"""
gen_instruct.py
This script will generate instruction-following examples from the cleaned data.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

# Config
BASE_DELAY = 1.0  # Base seconds between API calls
MAX_RETRIES = 3  # Maximum retries per request
OUTPUT_FILE = "../data/instruct/zig_instruct_data.jsonl"
INPUT_FILE = "../data/cleaned/zig_cleaned_data.jsonl"
MODEL = "tencent/hy3-preview:free"
API_BASE_URL = "https://openrouter.ai/api/v1"

load_dotenv()
openrouter_token = os.getenv("OPENROUTER_TOKEN")

if not openrouter_token:
    raise ValueError("OPENROUTER_TOKEN not found in environment variables")


def call_openrouter(code, max_retries=MAX_RETRIES):
    """Generate instruction for a Zig code snippet with retry logic."""
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
            wait_time = BASE_DELAY * (2**attempt)  # Exponential backoff
            print(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts")
                return None
    return None


def generate_instruction():
    """Generate instruction-following dataset from cleaned Zig code."""

    # Create output directory if not existing
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    total_processed = 0
    total_written = 0
    total_errors = 0

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(OUTPUT_FILE, "a", encoding="utf-8") as outfile,
    ):
        for line in infile:
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

            # Generate instruction with rate limiting
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
            total_written += 1

            total_processed += 1
            if total_processed % 10 == 0:
                print(f"Processed {total_processed} examples...")

            # Rate limiter
            time.sleep(BASE_DELAY)

    print("\n=== Generation Complete ===")
    print(f"Total processed: {total_processed}")
    print(f"Total written: {total_written}")
    print(f"Total errors: {total_errors}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_instruction()
