"""
Generates a lightweight datset of arithmetic Q→A pairs for the llm to train on.
"""

import argparse
import random
from pathlib import Path

# helpers
OPERATORS = [
    ("+", lambda a, b: a + b),
    ("-", lambda a, b: a - b),
    ("*", lambda a, b: a * b),
]

QUESTION_STYLES = [
    lambda a, op, b, ans: f"{a}{op}{b}={ans}",
    lambda a, op, b, ans: f"What is {a} {op} {b}? {ans}",
]


def make_example(rng: random.Random) -> str:
    """Generate a single math Q→A line."""
    op_sym, fn = rng.choice(OPERATORS)

    a = rng.randint(0, 99)
    b = rng.randint(0, 99)

    # avoid negative 
    if op_sym == "-" and a < b:
        a, b = b, a

    ans = fn(a, b)
    style = rng.choice(QUESTION_STYLES)
    return style(a, op_sym, b, ans)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate math QA corpus")
    parser.add_argument("--n", type=int, default=500_000,
                        help="number of examples to generate (default 100k)")
    parser.add_argument("--out", type=Path, default=Path("dataset.txt"),
                        help="output text file (one example per line)")
    parser.add_argument("--seed", type=int, default=0,
                        help="PRNG seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", encoding="utf-8") as f:
        for _ in range(args.n):
            f.write(make_example(rng) + "\n")

    print(f"Wrote {args.n:,} examples to {args.out}")
