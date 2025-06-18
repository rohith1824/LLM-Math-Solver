import numpy as np
from pathlib import Path
from typing import Generator, Tuple
from tokenizer import Tokenizer

class MathDataset:
    """Load TSV, build tokenizer, encode/pad, split, batch."""

    def __init__(
        self,
        txt_path: str = "data/math_dataset.txt",
        pad_to: int = 64,
        val_split: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.path = Path(txt_path)
        self.pad_to = pad_to
        self.val_split = val_split
        self.seed = seed
        self._prepare()

    def _prepare(self) -> None:
        # read raw lines → lists of strings
        lines = self.path.read_text("utf-8").splitlines()
        qs, ans = zip(*(ln.split("\t") for ln in lines))

        # build tokenizer from full corpus
        self.tok = Tokenizer(list(qs) + list(ans))

        # fetch special‑token IDs straight from vocab
        self.pad_id   = self.tok.token2id[self.tok.pad_token]
        self.start_id = self.tok.token2id[self.tok.start_token]
        self.end_id   = self.tok.token2id[self.tok.end_token]

        # encode and pad/truncate (encode() already includes <START>/<END>)
        enc_q = [self._pad(self.tok.encode(q)) for q in qs]
        enc_a = [self._pad(self.tok.encode(a)) for a in ans]

        X = np.array(enc_q, dtype=np.int32)
        Y = np.array(enc_a, dtype=np.int32)

        # 5. shuffle + split
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(X))
        split_at = int(len(X) * (1 - self.val_split))

        self.x_train, self.y_train = X[idx[:split_at]], Y[idx[:split_at]]
        self.x_val,   self.y_val   = X[idx[split_at:]], Y[idx[split_at:]]

    def _pad(self, ids: list[int]) -> list[int]:
        if len(ids) >= self.pad_to:
            return ids[: self.pad_to]
        return ids + [self.pad_id] * (self.pad_to - len(ids))

    def train_batches(
        self, batch_size: int = 32
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from self._batchify(self.x_train, self.y_train, batch_size)

    def val_batches(
        self, batch_size: int = 32
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        yield from self._batchify(self.x_val, self.y_val, batch_size)

    @staticmethod
    def _batchify(X: np.ndarray, Y: np.ndarray, batch_size: int):
        for i in range(0, len(X), batch_size):
            yield X[i : i + batch_size], Y[i : i + batch_size]


if __name__ == "__main__":
    ds = MathDataset()
    xb, yb = next(ds.train_batches(4))
    print("xb shape:", xb.shape, "yb shape:", yb.shape)
