"""
clean_data.py
This script will clean raw data from data/raw/ and output it to data/cleaned/
"""

import hashlib
import json
import os


def clean_data():
    input_file = "../data/raw/zig_raw_data.jsonl"
    output_file = "../data/cleaned/zig_cleaned_data.jsonl"

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    seen_hashes = set()
    total_read = 0
    duplicates_removed = 0
    too_short_removed = 0
    too_long_removed = 0
    non_ascii_removed = 0
    total_kept = 0

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if not line:
                continue

            total_read += 1

            # Parse JSON
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            content = record.get("content", "")

            # Deduplicate by content hash
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                duplicates_removed += 1
                continue
            seen_hashes.add(content_hash)

            # Check content length
            content_len = len(content)
            if content_len < 50:
                too_short_removed += 1
                continue
            if content_len > 8000:
                too_long_removed += 1
                continue

            # Check non-ASCII ratio
            non_ascii_count = sum(1 for c in content if ord(c) > 127)
            non_ascii_ratio = non_ascii_count / content_len
            if non_ascii_ratio > 0.20:
                non_ascii_removed += 1
                continue

            # Write record to output
            outfile.write(json.dumps(record) + "\n")
            total_kept += 1

    # Print final report
    print("=== Cleaning Report ===")
    print(f"Total read: {total_read}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Too short removed: {too_short_removed}")
    print(f"Too long removed: {too_long_removed}")
    print(f"Non-ASCII removed: {non_ascii_removed}")
    print(f"Total kept: {total_kept}")


if __name__ == "__main__":
    clean_data()
