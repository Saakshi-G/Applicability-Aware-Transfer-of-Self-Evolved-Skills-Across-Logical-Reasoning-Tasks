from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.dataset import load_bbh_task
from src.baseline import extract_answer, is_correct


TASK = "tracking_shuffled_objects_five_objects"
NUM_EXAMPLES = 10


def main():

    examples = load_bbh_task(TASK)

    examples = examples[:NUM_EXAMPLES]

    print(f"Task: {TASK}")
    print(f"Running baseline on {len(examples)} examples")

    for i, example in enumerate(examples, start=1):

        question = example["input"]
        target = example["target"]

        print("\n" + "=" * 60)
        print(f"Example {i}")
        print("=" * 60)

        print("\nQuestion:")
        print(question)

        print("\nGold answer:")
        print(target)


if __name__ == "__main__":
    main()
    