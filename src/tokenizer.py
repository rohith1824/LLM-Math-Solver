import re

class Tokenizer:
    """
    Right-to-left digit tokenizer for math LLMs.
    """
    def __init__(self, texts,
                 pad_token="<PAD>", start_token="<START>",
                 end_token="<END>", unk_token="<UNK>"):
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token   = end_token
        self.unk_token   = unk_token

        # Build vocabulary: digits 0-9 plus forced math symbols and corpus chars
        digits = set("0123456789")
        math_syms = set("+-*/=() ")
        # include everything seen in texts plus our math symbols
        all_chars = set("".join(texts)) | math_syms
        # separate non-digit chars and keep digits separate
        other = sorted(all_chars - digits)

        # tokens list: specials, other chars, then digits
        self.tokens = [pad_token, start_token, end_token, unk_token] + other + sorted(digits)

        # create id mappings
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}

    def _iter_units(self, text):
        """
        Yield units: digits reversed in multi-digit numbers, or single characters.
        """
        i = 0
        while i < len(text):
            if text[i].isdigit():
                j = i
                while j < len(text) and text[j].isdigit():
                    j += 1
                # emit digits right-to-left
                for d in reversed(text[i:j]):
                    yield d
                i = j
            else:
                yield text[i]
                i += 1

    def encode(self, text):
        ids = [self.token2id[self.start_token]]
        for unit in self._iter_units(text):
            ids.append(self.token2id.get(unit, self.token2id[self.unk_token]))
        ids.append(self.token2id[self.end_token])
        return ids

    def decode(self, ids, skip_special=True):
        # convert ids back to tokens
        toks = []
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if skip_special and tok in {self.pad_token, self.start_token,
                                        self.end_token, self.unk_token}:
                continue
            toks.append(tok)

        # restore each reversed digit group
        out, buf = [], []
        for t in toks:
            if t.isdigit():
                buf.append(t)
            else:
                if buf:
                    out.extend(reversed(buf))  # flip digits back
                    buf = []
                out.append(t)
        if buf:
            out.extend(reversed(buf))
        return "".join(out)


if __name__ == "__main__":
    samples = ["What is 123 + 45?", "Solve 8*7"]
    tok = Tokenizer(samples)

    seq = tok.encode("What is 123 + 45?")
    print("Encoded:", seq)
    print("Decoded:", tok.decode(seq))
