class SimpleTokenizer:
    """
    character-level tokenizer for LLMs.
    """

    def __init__(
        self,
        texts,
        pad_token: str = "<PAD>",
        start_token: str = "<START>",
        end_token: str = "<END>",
        unk_token: str = "<UNK>",
    ):
        self.pad_token   = pad_token
        self.start_token = start_token
        self.end_token   = end_token
        self.unk_token   = unk_token

        # Build vocabulary
        core = set("0123456789+-*/=() ")
        all_chars = core | set("".join(texts))

        # Specials first, then the rest
        self.tokens = [pad_token, start_token, end_token, unk_token] + sorted(all_chars)
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def encode(self, text: str) -> list[int]:
        """Return list of token-IDs, wrapped with <START> … <END>."""
        ids = [self.token2id[self.start_token]]
        for ch in text:
            ids.append(self.token2id.get(ch, self.token2id[self.unk_token]))
        ids.append(self.token2id[self.end_token])
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert IDs back to string, optionally dropping special tokens."""
        pieces = []
        specials = {self.pad_token, self.start_token, self.end_token, self.unk_token}
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if skip_special and tok in specials:
                continue
            pieces.append(tok)
        return "".join(pieces)

# Test
if __name__ == "__main__":
    samples = ["What is 123 + 45?", "Solve 8*7"]
    tok = SimpleTokenizer(samples)

    for s in ["4", "2+2=4", "10 + 20 = 30", "What is 123 + 45?"]:
        enc = tok.encode(s)
        dec = tok.decode(enc)
        print(f"{s!r:22} → {enc}")
        print("decoded:", dec)
        print()
