import argparse
import json
import re
from hashlib import sha1

HEB_RE = re.compile(r"[\u0590-\u05FF]")

def detect_lang(text: str) -> str:
    heb = sum(1 for ch in text if "\u0590" <= ch <= "\u05FF")
    alpha = sum(1 for ch in text if ch.isalpha())
    if alpha == 0:
        return "unknown"
    return "he" if heb / alpha >= 0.20 else "non-he"

def split_into_paragraphs(text: str) -> list[str]:
    # Keep paragraphs separated; good for Hebrew too
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras

def chunk_paragraphs(paras: list[str], max_chars: int, overlap_chars: int) -> list[str]:
    chunks = []
    cur = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        chunk = "\n".join(cur).strip()
        if chunk:
            chunks.append(chunk)
        # overlap: keep last N chars from current chunk
        if overlap_chars > 0 and chunk:
            tail = chunk[-overlap_chars:]
            cur = [tail]
            cur_len = len(tail)
        else:
            cur = []
            cur_len = 0

    for p in paras:
        # If a single paragraph is huge, split it
        if len(p) > max_chars:
            if cur:
                flush()
            start = 0
            while start < len(p):
                end = min(len(p), start + max_chars)
                part = p[start:end].strip()
                if part:
                    chunks.append(part)
                start = max(0, end - overlap_chars) if overlap_chars > 0 else end
            continue

        if cur_len + len(p) + 1 <= max_chars:
            cur.append(p)
            cur_len += len(p) + 1
        else:
            flush()
            cur.append(p)
            cur_len = len(p)

    flush()
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="cis_pages_clean.jsonl")
    ap.add_argument("--out", default="cis_chunks.jsonl")
    ap.add_argument("--max_chars", type=int, default=2400)       # ~500-700 tokens typical
    ap.add_argument("--overlap_chars", type=int, default=300)    # small overlap
    args = ap.parse_args()

    total_docs = 0
    total_chunks = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        with open(args.inp, "r", encoding="utf-8") as in_f:
            for line in in_f:
                r = json.loads(line)
                text = (r.get("text_clean") or "").strip()
                if not text:
                    continue

                paras = split_into_paragraphs(text)
                chunks = chunk_paragraphs(paras, args.max_chars, args.overlap_chars)

                url = r.get("url", "")
                title = r.get("title", "")
                lastmod = r.get("lastmod")
                src = r.get("source_sitemap")
                doc_lang = r.get("lang_guess") or detect_lang(text)

                # stable doc id (for debugging)
                doc_id = sha1(url.encode("utf-8")).hexdigest()[:12]

                for i, ch in enumerate(chunks):
                    chunk_id = f"{doc_id}_{i:03d}"
                    rec = {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "url": url,
                        "title": title,
                        "lastmod": lastmod,
                        "source_sitemap": src,
                        "lang": doc_lang,
                        "text": ch,
                    }
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_chunks += 1

                total_docs += 1

    print(f"Chunked {total_docs} docs into {total_chunks} chunks -> {args.out}")

if __name__ == "__main__":
    main()
