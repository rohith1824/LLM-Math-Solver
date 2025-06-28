import numpy as np
from model import TinyTransformer
from dataset_loader import MathDataset
import pickle

# Hyper-parameters
SEQ_LEN    = 64          # context length
BATCH_SIZE = 32
EPOCHS     = 20
LR         = 5e-3        
D_MODEL    = 64        

# Dataset & vocabulary
ds   = MathDataset(pad_to=SEQ_LEN)   
tok  = ds.tok
VOCAB = len(tok.tokens)

model = TinyTransformer(
    vocab_size=VOCAB,
    ctx_len   =SEQ_LEN,
    d_model   =D_MODEL,
)

# Training loop
for epoch in range(1, EPOCHS + 1):
    total_loss, n_batches = 0.0, 0
    for X, Y in ds.train_batches(BATCH_SIZE):   # X,Y : (B,SEQ_LEN)
        logits = model.forward(X)
        loss   = model.loss_and_backward(logits, Y)
        model.step(lr=LR, clip=5.0)                       # SGD update

        total_loss += loss
        n_batches  += 1

    avg = total_loss / n_batches
    print(f"Epoch {epoch}/{EPOCHS}  |  avg loss = {avg:.4f}")

# Save weights & whole model
with open("tiny_transformer.pkl", "wb") as f:
    pickle.dump(model, f)
print(" Model saved to tiny_transformer.pkl")
