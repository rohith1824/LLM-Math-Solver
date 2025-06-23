import argparse
import pickle
import numpy as np
from dataset_loader import MathDataset

def generate(prompt: str, model, tok, pad_to: int) -> str:
    """
    Encode the prompt, pad/truncate to pad_to, run one forward pass,
    argmax over vocab at each of pad_to positions, then decode.
    """
    # 1) encode the prompt (adds <START> and <END>)
    ids = tok.encode(prompt)
    # 2) pad or truncate to exactly pad_to tokens
    pad_id = tok.token2id[tok.pad_token]
    if len(ids) < pad_to:
        ids = ids + [pad_id] * (pad_to - len(ids))
    else:
        ids = ids[:pad_to]
    X = np.array([ids], dtype=np.int32)     # shape (1, pad_to)

    # 3) forward pass
    probs = model(X)                        # shape (1, pad_to, V)
    pred_ids = probs.argmax(axis=-1)[0]     # shape (pad_to,)

    # 4) decode the entire predicted sequence (skipping specials & pads)
    return tok.decode(pred_ids, skip_special=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="model.pkl",
                   help="Pickled MiniTransformer instance")
    p.add_argument("--prompt", required=True,
                   help="Math question, e.g. 'What is 3 + 5?'")
    args = p.parse_args()

    # 1) Load your trained model
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    # 2) Rebuild the tokenizer exactly as in training
    ds = MathDataset(pad_to=model.T)
    tok = ds.tok

    # 3) Generate & print
    answer = generate(args.prompt, model, tok, pad_to=ds.pad_to)
    print(f"{args.prompt} → {answer}")

if __name__ == "__main__":
    main()
