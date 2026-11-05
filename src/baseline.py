import re


def extract_answer(response):
    """
    Extract the final multiple-choice answer from the model response.
    """

    if response is None:
        return None

    response = response.upper()

    final_match = re.search(
        r"FINAL\s*ANSWER\s*:\s*\(([A-Z])\)",
        response
    )

    if final_match:
        return f"({final_match.group(1)})"

    matches = re.findall(r"\(([A-Z])\)", response)

    if matches:
        return f"({matches[-1]})"

    return None


def is_correct(prediction, target):
    """
    Compare model prediction with the BBH target.
    """

    if prediction is None:
        return False

    return prediction.strip().upper() == target.strip().upper()


if __name__ == "__main__":

    test_response = """
    After tracking all of the swaps, the object is currently
    in the third position. Therefore, the answer is (C).
    """

    prediction = extract_answer(test_response)

    print("Extracted:", prediction)
    print("Correct:", is_correct(prediction, "(C)"))
