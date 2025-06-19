import numpy as np
from model import MiniTransformer
from dataset_loader import MathDataset


#hyper-params
VOCAB      =  len(MathDataset(pad_to=1).tok.tokens)  # extract vocab size
SEQ_LEN    = 64
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 1e-3
CLIP       = 1.0
D_MODEL    = 128
D_FF       = 4 * D_MODEL
N_LAYERS   = 2
N_HEADS    = 2

# build model & collect params
model = MiniTransformer(VOCAB, SEQ_LEN, D_MODEL, D_FF, N_LAYERS, N_HEADS)

params, grads = [], []
def add(p, g):
    params.append(p); grads.append(g)

# embedding
add(model.embed.E, model.embed.E_grad)
# transformer blocks
for blk in model.blocks:
    add(blk.attn.Wq, blk.attn.Wq_grad); add(blk.attn.Wk, blk.attn.Wk_grad)
    add(blk.attn.Wv, blk.attn.Wv_grad); add(blk.attn.Wo, blk.attn.Wo_grad)
    add(blk.ff.W1 , blk.ff.W1_grad ); add(blk.ff.b1,  blk.ff.b1_grad)
    add(blk.ff.W2 , blk.ff.W2_grad ); add(blk.ff.b2,  blk.ff.b2_grad)
    add(blk.ln1.gamma, blk.ln1.gamma_grad); add(blk.ln1.beta, blk.ln1.beta_grad)
    add(blk.ln2.gamma, blk.ln2.gamma_grad); add(blk.ln2.beta, blk.ln2.beta_grad)
# output head
add(model.head.W, model.head.W_grad); add(model.head.b, model.head.b_grad)

# adam slots
m = [np.zeros_like(p) for p in params]
v = [np.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8

# helper functions
def soft_xent(probs, tgt):
    B, T, V = probs.shape
    idx = probs[np.arange(B)[:, None], np.arange(T)[None, :], tgt]
    loss = -np.log(idx + 1e-9).mean()
    grad = probs.copy()
    grad[np.arange(B)[:, None], np.arange(T)[None, :], tgt] -= 1
    grad /= (B * T)
    return loss, grad

def zero_grads():
    for g in grads:
        g.fill(0)

def ln_back(ln, dY):
    D = ln.gamma.shape[0]
    ln.gamma_grad += (dY * ln.xhat).sum((0, 1))
    ln.beta_grad  += dY.sum((0, 1))
    dxhat = dY * ln.gamma
    return (1/D)*ln.inv*(D*dxhat
                         - dxhat.sum(-1, keepdims=True)
                         - ln.xhat*(dxhat*ln.xhat).sum(-1, keepdims=True))

ds = MathDataset(pad_to=SEQ_LEN)  # uses data/math_dataset.txt
# ds.train_batches yields (X, Y) NumPy arrays of shape (BATCH_SIZE, SEQ_LEN)

# training loop
for ep in range(1, EPOCHS+1):
    total_loss = 0.0

    for X, Y in ds.train_batches(BATCH_SIZE):
        zero_grads()

        # forward
        P = model(X)                    # (BATCH, SEQ_LEN, VOCAB)
        loss, dlog = soft_xent(P, Y)
        total_loss += loss

        # head back-prop
        model.head.b_grad[:] = dlog.sum((0,1))
        model.head.W_grad[:] = model.head.x_in.reshape(-1, D_MODEL).T \
                               @ dlog.reshape(-1, VOCAB)
        dx = dlog @ model.head.W.T
        dx = dx.reshape(X.shape + (D_MODEL,))

        # blocks back-prop
        for blk in reversed(model.blocks):
            # LN2
            dx = ln_back(blk.ln2, dx)
            # FFN
            ff = blk.ff
            ff.W2_grad += ff.h.reshape(-1, D_FF).T @ dx.reshape(-1, D_MODEL)
            ff.b2_grad += dx.sum((0,1))
            dh = dx @ ff.W2.T
            dh[ff.h <= 0] = 0
            ff.W1_grad += blk.ln1.y.reshape(-1, D_MODEL).T \
                          @ dh.reshape(-1, ff.W1.shape[1])
            ff.b1_grad += dh.sum((0,1))
            dx = dh @ ff.W1.T
            # LN1
            dx = ln_back(blk.ln1, dx)
            # Attention (full grad)
            att = blk.attn
            B_, T_, h_, dh_ = att.att.shape[0], att.att.shape[2], att.h, att.dh

            # Wo grad
            att.Wo_grad += att.O_in.reshape(-1, D_MODEL).T \
                           @ dx.reshape(-1, D_MODEL)
            dC = dx @ att.Wo.T
            dHead = dC.reshape(B_, T_, h_, dh_).transpose(0,2,1,3)

            # Vh grad and Score grad
            Vh = att.Vh
            dVh = att.att.transpose(0,1,3,2) @ dHead
            dS  = dHead @ Vh.transpose(0,1,3,2)

            # softmax backward
            dAtt = dS * att.att - att.att * (dS * att.att).sum(-1, keepdims=True)

            # Qh, Kh grads
            scale = 1/np.sqrt(dh_)
            Qh, Kh = att.Qh, att.Kh
            dQh = (dAtt @ Kh) * scale
            dKh = (dAtt.transpose(0,1,3,2) @ Qh) * scale

            # merge back to D
            dQ = dQh.transpose(0,2,1,3).reshape(B_, T_, D_MODEL)
            dK = dKh.transpose(0,2,1,3).reshape(B_, T_, D_MODEL)
            dV = dVh.transpose(0,2,1,3).reshape(B_, T_, D_MODEL)

            # grads w.r.t Wq/Wk/Wv
            x_in = blk.x_in
            att.Wq_grad += x_in.reshape(-1, D_MODEL).T @ dQ.reshape(-1, D_MODEL)
            att.Wk_grad += x_in.reshape(-1, D_MODEL).T @ dK.reshape(-1, D_MODEL)
            att.Wv_grad += x_in.reshape(-1, D_MODEL).T @ dV.reshape(-1, D_MODEL)

            # propagate to next dx
            dx = dQ @ att.Wq.T + dK @ att.Wk.T + dV @ att.Wv.T

        # embedding grad
        np.add.at(model.embed.E_grad, X, dx)

        # Adam update
        for p, g, mi, vi in zip(params, grads, m, v):
            mi[:] = beta1 * mi + (1 - beta1) * g
            vi[:] = beta2 * vi + (1 - beta2) * (g**2)
            p -= LR * mi / (np.sqrt(vi) + eps)

    avg = total_loss / (ds.x_train.shape[0] / BATCH_SIZE)

    print(f"Epoch {ep}/{EPOCHS} | avg loss {avg:.4f}")
