import json
import os
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default location of the BBH clone (gitignored). Override on a cluster
# (e.g. ASU SOL) with:  export BBH_DIR=/path/to/Big-Bench-Hard/bbh
BBH_DIR = Path(
    os.environ.get(
        "BBH_DIR",
        PROJECT_ROOT / "external" / "Big-Bench-Hard" / "bbh",
    )
)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Each BBH family has 3 configs (3, 5, 7 objects) of 250 examples each.
# The proposal's "750 instances" is these three pooled.
CONFIGS = ("three_objects", "five_objects", "seven_objects")

# Split sizes from the proposal. Anything left over is reserved.
SPLIT_SIZES = {
    "logical_deduction": {"evolution": 150, "validation": 50, "test": 150},
    "tracking_shuffled_objects": {"dev": 50, "test": 150},
}


def load_bbh_task(task_name):
    """
    Load a task from the BIG-Bench Hard dataset.
    """

    file_path = BBH_DIR / f"{task_name}.json"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find BBH task: {file_path}\n"
            f"Clone the dataset into external/ with:\n"
            f"  git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git "
            f"external/Big-Bench-Hard\n"
            f"or set the BBH_DIR environment variable."
        )

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data["examples"]


def load_family(family):
    """
    Load all three configs of a BBH family (e.g. "logical_deduction") pooled
    into one list. Every example gets:
      - "id":     stable ID, "<family>_<config>:<index in BBH file>"
      - "config": which config it came from
    The ID depends only on the BBH file order, so it is the same on every
    machine and is what the saved splits and results files refer to.
    """

    pooled = []

    for config in CONFIGS:
        task_name = f"{family}_{config}"

        for index, example in enumerate(load_bbh_task(task_name)):
            pooled.append(
                {
                    "id": f"{task_name}:{index}",
                    "config": config,
                    "input": example["input"],
                    "target": example["target"],
                }
            )

    return pooled


def make_splits(family, seed=0):
    """
    Create seeded splits for a family, stratified by config so every split
    has the same 3/5/7-object mix. Splits are disjoint; unused examples go
    in "reserved". Returns {split_name: [example ids]}.
    """

    sizes = SPLIT_SIZES[family]
    examples = load_family(family)

    rng = random.Random(seed)

    by_config = {config: [] for config in CONFIGS}
    for example in examples:
        by_config[example["config"]].append(example["id"])

    for ids in by_config.values():
        ids.sort()
        rng.shuffle(ids)

    # Per-config quota for each split: as even as possible across configs.
    splits = {name: [] for name in sizes}
    splits["reserved"] = []

    cursor = {config: 0 for config in CONFIGS}

    for name, total in sizes.items():
        base, extra = divmod(total, len(CONFIGS))

        for i, config in enumerate(CONFIGS):
            quota = base + (1 if i < extra else 0)
            start = cursor[config]
            splits[name].extend(by_config[config][start:start + quota])
            cursor[config] = start + quota

    for config in CONFIGS:
        splits["reserved"].extend(by_config[config][cursor[config]:])

    return splits


def save_splits(family, seed=0):
    """
    Write data/splits/<family>.json. Commit this file so the whole team (and
    every experimental condition) evaluates on exactly the same examples.
    """

    splits = make_splits(family, seed=seed)

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLITS_DIR / f"{family}.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump({"seed": seed, "splits": splits}, file, indent=2)

    return path


def load_split(family, split_name):
    """
    Return the list of example dicts for one split of a family,
    using the saved split file.
    """

    path = SPLITS_DIR / f"{family}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.dataset  to create it."
        )

    with open(path, "r", encoding="utf-8") as file:
        splits = json.load(file)["splits"]

    wanted = set(splits[split_name])

    return [example for example in load_family(family) if example["id"] in wanted]


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

    # Create / refresh the saved splits (seed 0).
    for family in SPLIT_SIZES:
        path = save_splits(family, seed=0)
        print("Wrote", path)

    show_examples(
        "logical_deduction_five_objects",
        n=1
    )

    show_examples(
        "tracking_shuffled_objects_five_objects",
        n=1
    )
