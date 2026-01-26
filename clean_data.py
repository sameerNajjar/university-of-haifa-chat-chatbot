import argparse
import json
import re
from collections import Counter

HEB_RE = re.compile(r"[\u0590-\u05FF]")

def normalize_text(text: str) -> str:
    # Normalize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim spaces on each line
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join(lines).strip()

def line_is_noise(line: str) -> bool:
    if not line:
        return True
    # too short (often menu items)
    if len(line) < 3:
        return True
    # very short lines that contain no Hebrew and no digits (often nav)
    if len(line) < 12 and not HEB_RE.search(line) and not any(ch.isdigit() for ch in line):
        return True
    return False

def dedupe_lines(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for ln in lines:
        if ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="cis_pages.jsonl")
    ap.add_argument("--out", default="cis_pages_clean.jsonl")
    ap.add_argument("--min_chars", type=int, default=400)

    # Boilerplate removal: lines that appear in >= this fraction of documents
    ap.add_argument("--boiler_frac", type=float, default=0.20)
    ap.add_argument("--boiler_min_len", type=int, default=8)
    args = ap.parse_args()

    records = []
    line_doc_freq = Counter()

    # 1) load + compute doc-frequency of lines
    with open(args.inp, "r", encoding="utf-8") as f:
        for raw in f:
            r = json.loads(raw)
            text = normalize_text(r.get("text", ""))
            lines = [ln for ln in text.split("\n") if ln.strip()]
            # count each unique line once per doc
            uniq = set(ln for ln in lines if len(ln) >= args.boiler_min_len)
            for ln in uniq:
                line_doc_freq[ln] += 1
            r["_lines"] = lines
            records.append(r)

    n_docs = max(1, len(records))
    boiler_threshold = int(n_docs * args.boiler_frac)

    boiler_lines = {ln for ln, df in line_doc_freq.items() if df >= boiler_threshold}

    # 2) clean each record
    saved = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for r in records:
            lines = r.pop("_lines", [])
            # remove boilerplate
            lines = [ln for ln in lines if ln not in boiler_lines]
            # remove noisy lines
            lines = [ln for ln in lines if not line_is_noise(ln)]
            # dedupe inside doc
            lines = dedupe_lines(lines)

            cleaned = "\n".join(lines).strip()
            if len(cleaned) < args.min_chars:
                continue

            r["text_clean"] = cleaned
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            saved += 1

    print(f"Saved {saved} cleaned docs to {args.out}")
    print(f"Removed {len(boiler_lines)} boilerplate lines (threshold: {boiler_threshold}/{n_docs} docs)")

if __name__ == "__main__":
    main()
