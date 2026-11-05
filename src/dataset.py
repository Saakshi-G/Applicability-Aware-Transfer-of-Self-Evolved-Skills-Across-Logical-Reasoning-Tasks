import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BBH_DIR = PROJECT_ROOT / "external" / "Big-Bench-Hard" / "bbh"


def load_bbh_task(task_name):
    """
    Load a task from the BIG-Bench Hard dataset.
    """

    file_path = BBH_DIR / f"{task_name}.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find BBH task: {file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["examples"]


def show_examples(task_name, n=3):
    examples = load_bbh_task(task_name)

    print(f"\nTask: {task_name}")
    print(f"Number of examples: {len(examples)}")
    print("=" * 60)

    for i, example in enumerate(examples[:n], start=1):

        print(f"\nExample {i}")

        print("\nQUESTION:")
        print(example["input"])

        print("\nCORRECT ANSWER:")
        print(example["target"])

        print("-" * 60)


if __name__ == "__main__":

    show_examples(
        "logical_deduction_five_objects",
        n=3
    )

    show_examples(
        "tracking_shuffled_objects_five_objects",
        n=3
    )