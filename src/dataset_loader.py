"""
Turn a plain-text corpus of Q→A pairs into padded NumPy arrays ready for
training.
"""

from pathlib import Path
from typing   import Generator, Tuple, List
import re, numpy as np
from tokenizer import SimpleTokenizer

class MathDataset:
    def __init__(
        self,
        txt_path : str  = "data/dataset.txt",
        pad_to   : int  = 64,
        val_split: float= 0.1,
        seed     : int  = 42,
    ) -> None:
        self.path      = Path(txt_path)
        self.pad_to    = pad_to
        self.val_split = val_split
        self.seed      = seed
        self._prepare()

    # helpers functions
    @staticmethod
    def _split_line(line: str) -> Tuple[str, str]:
        """Return (question, answer) from one raw corpus line."""
        line = line.strip()
        if "\t" in line:                       # old TSV
            q, a = line.split("\t", 1)
            return q, a

        if "=" in line:                        # "expr=42"
            q, a = line.rsplit("=", 1)         # last '='
            return q + "=", a                  # keep '=' in the prompt

        # fallback
        m = re.match(r"^(.*\S)\s+(\S+)$", line)
        if not m:
            raise ValueError(f"Cannot parse line: {line}")
        return m.group(1), m.group(2)

    def _pad(self, ids: List[int]) -> List[int]:
        """Pad / truncate to fixed length."""
        if len(ids) >= self.pad_to:
            return ids[: self.pad_to]
        pad_id = self.tok.token2id[self.tok.pad_token]
        return ids + [pad_id] * (self.pad_to - len(ids))

    # corpus → arrays
    def _prepare(self) -> None:
        raw_lines = self.path.read_text("utf-8").splitlines()
        qs, ans   = zip(*(self._split_line(ln) for ln in raw_lines))

        # tokenizer on whole text
        self.tok = SimpleTokenizer(list(qs) + list(ans))

        # encode & pad
        enc_q = [self._pad(self.tok.encode(q)) for q in qs]
        enc_a = [self._pad(self.tok.encode(a)) for a in ans]
        X = np.array(enc_q, dtype=np.int32)
        Y = np.array(enc_a, dtype=np.int32)

        # 3shuffle / split
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        split = int(len(X) * (1 - self.val_split))

        self.x_train, self.y_train = X[idx[:split]], Y[idx[:split]]
        self.x_val,   self.y_val   = X[idx[split:]], Y[idx[split:]]

    # batch generators
    @staticmethod
    def _batchify(X: np.ndarray, Y: np.ndarray, bs: int):
        for i in range(0, len(X), bs):
            yield X[i : i + bs], Y[i : i + bs]

    def train_batches(
        self, batch_size: int = 32
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from self._batchify(self.x_train, self.y_train, batch_size)

    def val_batches(
        self, batch_size: int = 32
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from self._batchify(self.x_val, self.y_val, batch_size)

# test
if __name__ == "__main__":
    ds = MathDataset(pad_to=64)         
    xb, yb = next(ds.train_batches(4))
    print("xb", xb.shape, "->", yb.shape)
    print("decoded sample:", ds.tok.decode(xb[0], skip_special=True), "=>",
          ds.tok.decode(yb[0], skip_special=True))
