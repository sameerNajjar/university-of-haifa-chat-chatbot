import argparse
import json
import numpy as np
from sentence_transformers import SentenceTransformer

def read_meta(path: str):
    metas = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="cis_emb.npy")
    ap.add_argument("--meta", default="cis_meta.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--query", required=True)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    E = np.load(args.emb)  # normalized embeddings
    metas = read_meta(args.meta)
    assert len(metas) == E.shape[0], "meta rows must match embeddings rows"

    model = SentenceTransformer(args.model)
    q = model.encode(["query: " + args.query], normalize_embeddings=True)[0].astype(np.float32)

    # cosine similarity since normalized => dot product
    sims = E @ q
    topk = min(args.topk, len(sims))
    idx = np.argpartition(-sims, topk - 1)[:topk]
    idx = idx[np.argsort(-sims[idx])]

    for rank, i in enumerate(idx, start=1):
        m = metas[i]
        print(f"\n[{rank}] score={sims[i]:.4f}")
        print(f"URL:   {m.get('url')}")
        if m.get("title"):
            print(f"Title: {m.get('title')}")
        print("Text:")
        print(m.get("text", "")[:900], "..." if len(m.get("text","")) > 100 else "")

if __name__ == "__main__":
    main()
