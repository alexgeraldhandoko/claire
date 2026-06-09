from pathlib import Path
from collections import Counter
import json

FILE_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "onedaybtc.data"

types = Counter()
topics = Counter()
num_of_lines = 0
num_of_json_lines = 0
num_of_non_json_lines = 0
reset_line_numbers = []

with FILE_PATH.open("r") as file:
    for line_number, line in enumerate(file, start=1):
        stripped_line = line.strip()
        num_of_lines += 1

        try: 
            json_line = json.loads(stripped_line)
            num_of_json_lines += 1
            types[json_line["type"]] += 1
            topics[json_line["topic"]] += 1
            if json_line["type"] == "snapshot":
                reset_line_numbers.append(line_number)
        except json.JSONDecodeError as error:
            num_of_non_json_lines += 1
            print("This line is not JSON")
    
    print(f"Total number of lines: {num_of_lines}")
    print()
    print(f"Number of JSON lines: {num_of_json_lines}")
    print(f"Number of non-JSON lines: {num_of_non_json_lines}")
    print()
    print(f"Most common type: {types.most_common()}")
    print()
    print(f"Most common topic: {topics.most_common()}")
    print(f"Reset happens at line numbers: {reset_line_numbers}")