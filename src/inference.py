import argparse, pickle, numpy as np
from dataset_loader import MathDataset

def generate(prompt: str, model, tok):
    ids = tok.encode(prompt.strip())              # <START> … <END>
    pad = tok.token2id[tok.pad_token]

    feed = (ids + [pad] * model.T)[: model.T]
    logits = model.forward(np.array([feed], dtype=np.int32))[0]   # (T,V)
    preds  = logits.argmax(-1)                                    # (T,)

    specials = {
        tok.token2id[tok.start_token],    
        tok.token2id[tok.pad_token],
        tok.token2id[tok.end_token],
        tok.token2id[tok.unk_token],
    }

    # strip leading specials,  keepreading until first PAD / END
    answer_ids = [i for i in preds if i not in specials]

    ids   = tok.encode("17+8=")[:-1]
    feed  = (ids + [tok.token2id[tok.pad_token]]*model.T)[:model.T]
    preds = model.forward(np.array([feed]))[0].argmax(-1)
    print([tok.id2token[i] for i in preds[:10]])

    return tok.decode(answer_ids, skip_special=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",  default="tiny_transformer.pkl")
    ap.add_argument("--prompt", required=True)
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    tok = MathDataset(pad_to=model.T).tok
    print(f"{args.prompt} → {generate(args.prompt, model, tok)}")

if __name__ == "__main__":
    main()
