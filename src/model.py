import numpy as np
from typing import List

# Rotary helpers
def rotary_angles(T: int, d: int):
    inv = 1.0 / (10000 ** (np.arange(0, d, 2, dtype=np.float32) / d))
    pos = np.arange(T, dtype=np.float32)
    freqs = np.outer(pos, inv)
    return np.cos(freqs), np.sin(freqs)

def rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, :, :], sin[None, :, :]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out

# Core layers
class Embedding:
    def __init__(self, V, D):
        self.E = np.random.randn(V, D).astype(np.float32)*0.02
        self.E_grad = np.zeros_like(self.E)
    def __call__(self, ids):
        self.last_ids = ids
        return self.E[ids]

class LayerNorm:
    def __init__(self, D, eps=1e-5):
        self.gamma = np.ones(D, dtype=np.float32)
        self.beta  = np.zeros(D, dtype=np.float32)
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad  = np.zeros_like(self.beta)
        self.eps = eps
    def __call__(self, x):
        self.x    = x
        self.mu   = x.mean(-1, keepdims=True)
        self.var  = x.var(-1, keepdims=True)
        self.inv  = 1.0/np.sqrt(self.var + self.eps)
        self.xhat = (x - self.mu)*self.inv
        self.y    = self.gamma*self.xhat + self.beta
        return self.y

class MHSA:
    def __init__(self, D, h=2):
        assert D % h == 0
        self.D, self.h, self.dh = D, h, D//h
        self.Wq = np.random.randn(D,D).astype(np.float32)*0.02
        self.Wk = np.random.randn(D,D).astype(np.float32)*0.02
        self.Wv = np.random.randn(D,D).astype(np.float32)*0.02
        self.Wo = np.random.randn(D,D).astype(np.float32)*0.02
        self.Wq_grad = np.zeros_like(self.Wq)
        self.Wk_grad = np.zeros_like(self.Wk)
        self.Wv_grad = np.zeros_like(self.Wv)
        self.Wo_grad = np.zeros_like(self.Wo)
        self.cache = None

    @staticmethod
    def _mask(T):
        return np.triu(np.ones((T,T),dtype=np.float32),k=1)*-1e9

    def __call__(self, x):
        B,T,_ = x.shape
        self.x_in = x                    # for Wq/Wk/Wv grads
        self.Q = x @ self.Wq             # (B,T,D)
        self.K = x @ self.Wk
        self.V = x @ self.Wv

        # split to heads
        Qh = self.Q.reshape(B,T,self.h,self.dh).transpose(0,2,1,3)  # (B,h,T,dh)
        Kh = self.K.reshape(B,T,self.h,self.dh).transpose(0,2,1,3)
        Vh = self.V.reshape(B,T,self.h,self.dh).transpose(0,2,1,3)

        # cache for back-prop
        self.Qh, self.Kh, self.Vh = Qh, Kh, Vh

        # rotary
        if self.cache is None or self.cache[0].shape[0] < T:
            self.cache = rotary_angles(T, self.dh)
        cos,sin = self.cache
        Qh = rope(Qh, cos[:T], sin[:T])
        Kh = rope(Kh, cos[:T], sin[:T])

        # attention
        S = (Qh @ Kh.transpose(0,1,3,2)) / np.sqrt(self.dh)
        S = S + self._mask(T)
        expS = np.exp(S - S.max(-1,keepdims=True))
        self.att = expS/expS.sum(-1,keepdims=True)               # (B,h,T,T)

        heads = self.att @ Vh                                     # (B,h,T,dh)
        concat = heads.transpose(0,2,1,3).reshape(B,T,self.D)     # (B,T,D)
        self.O_in = concat                                        # cache
        return concat @ self.Wo                                   # (B,T,D)

class FeedForward:
    def __init__(self, D, Dff):
        self.W1 = np.random.randn(D,Dff).astype(np.float32)*0.02
        self.b1 = np.zeros(Dff, dtype=np.float32)
        self.W2 = np.random.randn(Dff,D).astype(np.float32)*0.02
        self.b2 = np.zeros(D, dtype=np.float32)
        self.W1_grad = np.zeros_like(self.W1)
        self.b1_grad = np.zeros_like(self.b1)
        self.W2_grad = np.zeros_like(self.W2)
        self.b2_grad = np.zeros_like(self.b2)
    def __call__(self, x):
        self.x_in = x
        self.h = np.maximum(0, x @ self.W1 + self.b1)
        return self.h @ self.W2 + self.b2

class Block:
    def __init__(self, D, Dff, h):
        self.attn = MHSA(D,h)
        self.ln1  = LayerNorm(D)
        self.ff   = FeedForward(D,Dff)
        self.ln2  = LayerNorm(D)
    def __call__(self, x):
        self.x_in = x
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class OutputHead:
    def __init__(self, D, V):
        self.W = np.random.randn(D,V).astype(np.float32)*0.02
        self.b = np.zeros(V, dtype=np.float32)
        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)
    def __call__(self, x):
        self.x_in = x
        L = x @ self.W + self.b
        L = L - L.max(-1,keepdims=True)
        E = np.exp(L)
        return E/E.sum(-1,keepdims=True)

class MiniTransformer:
    def __init__(self, V, T, D=128, Dff=512, L=2, H=2):
        self.embed  = Embedding(V,D)
        self.blocks: List[Block] = [Block(D,Dff,H) for _ in range(L)]
        self.head   = OutputHead(D,V)
        self.T = T
    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x)
    __call__ = forward
