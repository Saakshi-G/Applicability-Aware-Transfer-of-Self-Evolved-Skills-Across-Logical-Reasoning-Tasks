from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_bbh_task
from src.baseline import build_prompt, extract_answer, is_correct
from src.llm import get_llm


TASK = "logical_deduction_seven_objects"
N = 20


def main():
    examples = load_bbh_task(TASK)[:N]

    llm = get_llm()

    correct = 0
    parse_failures = 0

    for i, example in enumerate(examples, start=1):
        response = llm(
            build_prompt(example["input"])
        )

        prediction = extract_answer(response)

        if prediction is None:
            parse_failures += 1

        result = is_correct(
            prediction,
            example["target"]
        )

        if result:
            correct += 1

        print(
            f"{i:02d} | "
            f"pred={prediction} | "
            f"gold={example['target']} | "
            f"{'CORRECT' if result else 'WRONG'}"
        )

    print("\n==============================")
    print("MODEL SCREENING RESULT")
    print("==============================")

    print("Model: Gemini 3.5 Flash-Lite")
    print("Task:", TASK)
    print("Examples:", N)
    print("Correct:", correct)
    print("Incorrect:", N - correct)
    print("Accuracy:", round(correct / N, 3))
    print("Parse failures:", parse_failures)


if __name__ == "__main__":
    main()