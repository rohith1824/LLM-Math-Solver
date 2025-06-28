import numpy as np

DEBUG = False  # True prints gradient std every step

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x, dtype=np.float32)
    return ex / ex.sum(axis=axis, keepdims=True)


class TinyTransformer:
    """single-layer, single-head causal self-attention in pure NumPy"""

    def __init__(self, vocab_size: int, ctx_len: int, d_model: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V, self.T, self.D = vocab_size, ctx_len, d_model

        # parameters
        self.E  = rng.standard_normal((self.V, d_model), dtype=np.float32) * 0.02
        self.Wq = rng.standard_normal((d_model, d_model), dtype=np.float32) * 0.02
        self.Wk = rng.standard_normal((d_model, d_model), dtype=np.float32) * 0.02
        self.Wv = rng.standard_normal((d_model, d_model), dtype=np.float32) * 0.02
        self.Wo = rng.standard_normal((d_model, vocab_size), dtype=np.float32) * 0.02

        # grad buffers
        self.grads = {n: np.zeros_like(p) for n, p in {
            "E": self.E, "Wq": self.Wq, "Wk": self.Wk,
            "Wv": self.Wv, "Wo": self.Wo}.items()}

        # causal mask and scaling
        self.mask  = np.triu(np.ones((ctx_len, ctx_len), dtype=np.float32), k=1) * -1e9
        self.scale = 1.0 / np.sqrt(d_model)

    # forward
    def forward(self, ids: np.ndarray) -> np.ndarray:
        """ids: (B, T) int32  ->  logits: (B, T, V)"""
        self.ids = ids
        B, T = ids.shape

        self.X  = self.E[ids]              # (B,T,D)
        self.Q  = self.X @ self.Wq
        self.K  = self.X @ self.Wk
        self.Vv = self.X @ self.Wv

        att = (self.Q @ self.K.transpose(0, 2, 1)) * self.scale
        att += self.mask[:T, :T]
        self.att_w = softmax(att, axis=-1)         # (B,T,T)

        self.context = self.att_w @ self.Vv        # (B,T,D)
        logits = self.context @ self.Wo            # (B,T,V)
        return logits

    # loss + backward
    def loss_and_backward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        B, T, V = logits.shape

        probs = softmax(logits, axis=-1)

        # one-hot scatter
        onehot = np.zeros_like(probs, dtype=np.float32)
        rows = np.repeat(np.arange(B), T)
        cols = np.tile(np.arange(T), B)
        onehot[rows, cols, labels.ravel()] = 1.0

        loss = -np.log((probs * onehot).sum(-1) + 1e-9).mean()

        dlogits = (probs - onehot) / (B * T)
        if DEBUG: print("dlogits std:", dlogits.std())

        # Wo gradients
        self.grads["Wo"][...] = self.context.reshape(-1, self.D).T @ dlogits.reshape(-1, V)
        dcontext = dlogits @ self.Wo.T                        # (B,T,D)

        # Backprop through attention
        datt = dcontext @ self.Vv.transpose(0, 2, 1)          # (B,T,T)
        dVv = self.att_w.transpose(0, 2, 1) @ dcontext        # (B,T,D)

        # softmax backprop
        dS = self.att_w * (datt - (datt * self.att_w).sum(-1, keepdims=True))
        dS = dS * self.scale

        dQ = dS @ self.K
        dK = dS.transpose(0, 2, 1) @ self.Q

        # Parameter grads
        self.grads["Wq"][...] = self.X.reshape(-1, self.D).T @ dQ.reshape(-1, self.D)
        self.grads["Wk"][...] = self.X.reshape(-1, self.D).T @ dK.reshape(-1, self.D)
        self.grads["Wv"][...] = self.X.reshape(-1, self.D).T @ dVv.reshape(-1, self.D)

        # dX accumulates contributions
        dX = (dQ @ self.Wq.T) + (dK @ self.Wk.T) + (dVv @ self.Wv.T)

        # Embedding grads
        gradE = np.zeros_like(self.E)
        idx = self.ids.ravel()
        np.add.at(gradE, idx, dX.reshape(-1, self.D))
        self.grads["E"][...] = gradE

        return loss
    
    # SGD update
    def step(self, lr: float = 1e-3, clip: float = 1.0):
        for name, param in [("E", self.E), ("Wq", self.Wq), ("Wk", self.Wk),
                            ("Wv", self.Wv), ("Wo", self.Wo)]:
            g = self.grads[name]
            np.clip(g, -clip, clip, out=g)
            if DEBUG: print(f"∇{name} std:", g.std())
            param -= lr * g
            g.fill(0.0)


# test
if __name__ == "__main__":
    tok_ids = {"2": 26, "+": 8, "=": 11, "4": 28}
    seq_in  = np.array([[tok_ids["2"], tok_ids["+"], tok_ids["2"], tok_ids["="]]], dtype=np.int32)
    seq_lab = np.array([[tok_ids["+"], tok_ids["2"], tok_ids["="], tok_ids["4"]]], dtype=np.int32)

    model = TinyTransformer(vocab_size=64, ctx_len=4, d_model=32)
    for step in range(601):
        logits = model.forward(seq_in)
        loss   = model.loss_and_backward(logits, seq_lab)
        model.step(lr=1e-3)
        if step % 100 == 0:
            print(f"step {step:4d} | loss {loss:.4f}")

    #next token
    nxt = model.forward(seq_in)[0, -1].argmax()
    print("predicted next-id:", nxt, "(should be 28 for '4')")
