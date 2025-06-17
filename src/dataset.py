import argparse
import pathlib
import re

from datasets import load_dataset, disable_progress_bar
from tqdm import tqdm

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def build_dataset(
    modules: list[str],
    per_module: int,
    out_path: pathlib.Path,
    seed: int,
):
    disable_progress_bar()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = per_module * len(modules)

    with out_path.open("w", encoding="utf-8") as f:
        for mod in modules:
            print(f"▶ sampling {per_module:,} examples from {mod}")
            # stream so you don’t download 2GB
            ds = load_dataset("deepmind/math_dataset", mod, split="train", streaming=True)
            it = ds.shuffle(buffer_size=10_000, seed=seed).__iter__()
            for _ in tqdm(range(per_module), desc=mod):
                ex = next(it)
                q = _clean(ex["question"])
                a = _clean(ex["answer"])
                f.write(f"{q}\t{a}\n")

    print(f"✅ wrote {total:,} Q-A pairs → {out_path.resolve()}")

def _parse_args():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--modules", nargs="+",
        default=["arithmetic__add_or_sub", "arithmetic__mul", "arithmetic__div"],
        help="DeepMind modules to include",
    )
    p.add_argument(
        "--per", type=int, default=10_000,
        help="Examples per module",
    )
    p.add_argument(
        "--out", default="data/math_dataset.txt",
        help="Output TSV file",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Shuffle seed",
    )
    return p.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    build_dataset(
        modules=args.modules,
        per_module=args.per,
        out_path=pathlib.Path(args.out),
        seed=args.seed,
    )
