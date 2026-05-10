"""
gen_instruct.py
This script will generate instruction-following examples from the cleaned data.

Supports resuming — if the script crashes, re-running it will skip
already-processed examples and continue from where it left off.

Uses multiple free models in rotation. On a 429, the rate-limited model
is cooled down and the next available model is used immediately.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()
openrouter_token = os.getenv("OPENROUTER_TOKEN")

# Config
MAX_SAMPLES = 10_000  # Target number of instruction examples to generate
BASE_DELAY = 2.0  # Seconds between every successful API call
MAX_RETRIES = 5  # Max retries before skipping an example entirely
RATELIMIT_WAIT = 60  # Default seconds to wait on 429 if no Retry-After header
OUTPUT_FILE = "../data/instruct/zig_instruct_data.jsonl"
INPUT_FILE = "../data/cleaned/zig_cleaned_data.jsonl"
API_BASE_URL = "https://openrouter.ai/api/v1"

# Models to rotate through — switches to next on 429
MODELS = [
    "baidu/cobuddy:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "poolside/laguna-m.1:free",
]

if not openrouter_token:
    raise ValueError("OPENROUTER_TOKEN not found in environment variables")


class ModelRotator:
    """
    Rotates through a list of models.
    Tracks per-model cooldown times so a rate-limited model is
    skipped until its cooldown expires.
    """

    def __init__(self, models):
        self.models = models
        self.cooldowns = {m: 0 for m in models}
        self.index = 0

    def get_next(self):
        """Return the next available model, cycling through the list."""
        now = time.time()
        for _ in range(len(self.models)):
            model = self.models[self.index % len(self.models)]
            self.index += 1
            if self.cooldowns[model] <= now:
                return model
        # All models are on cooldown — wait for the soonest one
        soonest_model = min(self.cooldowns, key=self.cooldowns.get)
        wait = max(0, self.cooldowns[soonest_model] - now)
        print(f"All models rate-limited. Waiting {wait:.0f}s for [{soonest_model}]...")
        time.sleep(wait)
        return soonest_model

    def cooldown(self, model, seconds):
        """Put a model on cooldown for a given number of seconds."""
        self.cooldowns[model] = time.time() + seconds
        print(f"  [{model}] on cooldown for {seconds}s.")


def load_already_processed(output_file):
    """Load code strings already written to output for resume support."""
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


def call_openrouter(code, rotator, max_retries=MAX_RETRIES):
    """
    Generate an instruction for a Zig code snippet.
    Rotates models on 429. Exponential backoff on other errors.
    """
    for attempt in range(max_retries):
        model = rotator.get_next()
        if attempt > 0:
            print(f"  Using model: [{model}]")

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
                    "model": model,
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

            # 429 — rate limited, cool down this model and try the next immediately
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                cooldown_time = int(retry_after) if retry_after else RATELIMIT_WAIT
                print(f"Rate limited (429) on [{model}].")
                rotator.cooldown(model, cooldown_time)
                continue  # No sleep — switch to next model immediately

            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            return result

        except requests.exceptions.RequestException as e:
            wait_time = BASE_DELAY * (2**attempt)  # 2s, 4s, 8s...
            print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts, skipping example.")
                return None

    return None


def generate_instruction():
    """Generate instruction-following dataset from cleaned Zig code."""
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    rotator = ModelRotator(MODELS)
    already_processed = load_already_processed(OUTPUT_FILE)
    already_done = len(already_processed)

    total_skipped = 0
    total_written = 0
    total_errors = 0

    print(f"Models in rotation: {MODELS}")
    print(f"Target: {MAX_SAMPLES} examples\n")

    with (
        open(INPUT_FILE, "r", encoding="utf-8") as infile,
        open(OUTPUT_FILE, "a", encoding="utf-8") as outfile,
    ):
        for line in infile:
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

            if code in already_processed:
                total_skipped += 1
                continue

            instruction = call_openrouter(code, rotator)
            if instruction is None:
                total_errors += 1
                continue

            example = {
                "instruction": instruction,
                "code": code,
                "repo_name": data.get("repo_name", ""),
                "path": data.get("path", ""),
                "language": data.get("language", "Zig"),
            }
            outfile.write(json.dumps(example) + "\n")
            outfile.flush()
            total_written += 1

            if total_written % 10 == 0:
                total_so_far = already_done + total_written
                print(f"Progress: {total_so_far}/{MAX_SAMPLES} examples written...")

            time.sleep(BASE_DELAY)

    print("\n=== Generation Complete ===")
    print(f"Target:           {MAX_SAMPLES}")
    print(f"Already resumed:  {already_done}")
    print(f"Newly written:    {total_written}")
    print(f"Total in output:  {already_done + total_written}")
    print(f"Errors:           {total_errors}")
    print(f"Output:           {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_instruction()
