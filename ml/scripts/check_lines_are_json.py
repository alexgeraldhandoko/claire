import json
from pathlib import Path

FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "onedaybtc.data"

total_lines = 0
total_json_lines = 0
total_non_json_lines = 0

with FILE_PATH.open("r") as file:
    for line in file:
        total_lines += 1

        stripped_line = line.strip()

        try:
            json.loads(stripped_line)
            total_json_lines += 1
        except json.JSONDecodeError as error:
            total_non_json_lines += 1

print(f"Total lines: {total_lines}")
print(f"Total JSON lines: {total_json_lines}")
print(f"Total non-JSON lines: {total_non_json_lines}")

