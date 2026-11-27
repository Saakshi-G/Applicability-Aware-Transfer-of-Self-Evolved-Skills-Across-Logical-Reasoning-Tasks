"""
Small deterministic sampling helper.

BBH examples are pooled in the order 3-object, 5-object, 7-object, so taking
"the first N" (what --limit-evolution did) gives only 3-object questions, the
easiest configuration. `subset` keeps the 3/5/7 mix instead.
"""

import random


def subset(examples, n, seed=0):
    """
    Return `n` examples, spread as evenly as possible across the "config"
    field (three_objects / five_objects / seven_objects), chosen with a fixed
    seed. The result keeps the original order, so it is the same on every
    machine. If n is None or >= len(examples), all examples are returned.
    """
    if n is None or n >= len(examples):
        return list(examples)

    rng = random.Random(seed)

    groups = {}
    for example in examples:
        groups.setdefault(example["config"], []).append(example)
    for group in groups.values():
        rng.shuffle(group)

    chosen = []
    while len(chosen) < n:
        progressed = False
        for config in sorted(groups):
            if groups[config] and len(chosen) < n:
                chosen.append(groups[config].pop())
                progressed = True
        if not progressed:
            break

    keep = {example["id"] for example in chosen}
    return [example for example in examples if example["id"] in keep]
