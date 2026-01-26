import argparse
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="cis_chunks.jsonl")
    ap.add_argument("--out_emb", default="cis_emb.npy")
    ap.add_argument("--out_meta", default="cis_meta.jsonl")
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    # Load chunks
    chunks = list(read_jsonl(args.inp))
    if not chunks:
        raise SystemExit("No chunks found. Check cis_chunks.jsonl")

    # E5 expects prefixes: "passage: " for docs, "query: " for queries
    texts = ["passage: " + c["text"] for c in chunks]

    model = SentenceTransformer(args.model)

    embs = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Embedding"):
        batch = texts[i:i+args.batch_size]
        e = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        embs.append(e)

    E = np.vstack(embs).astype(np.float32)
    np.save(args.out_emb, E)

    # Save metadata aligned with embeddings rows
    with open(args.out_meta, "w", encoding="utf-8") as f:
        for c in chunks:
            meta = {
                "chunk_id": c.get("chunk_id"),
                "doc_id": c.get("doc_id"),
                "url": c.get("url"),
                "title": c.get("title"),
                "lastmod": c.get("lastmod"),
                "lang": c.get("lang"),
                "text": c.get("text"),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"Saved embeddings: {args.out_emb}  shape={E.shape}")
    print(f"Saved metadata:   {args.out_meta}  rows={len(chunks)}")

if __name__ == "__main__":
    main()
