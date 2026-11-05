from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_bbh_task


task = "tracking_shuffled_objects_five_objects"

examples = load_bbh_task(task)

print("Number of examples:", len(examples))

example = examples[0]

print("\nKeys:")
print(example.keys())

print("\nINPUT:")
print(example["input"])

print("\nTARGET:")
print(example["target"])
