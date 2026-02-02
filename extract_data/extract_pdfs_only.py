import argparse
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Prefer pdfplumber if installed (often better), fallback to pypdf
try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except Exception:
    HAVE_PDFPLUMBER = False

from pypdf import PdfReader


PDF_RE = re.compile(r"\.pdf(\?|$)", re.IGNORECASE)


def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def to_https(url: str) -> str:
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def fetch(session: requests.Session, url: str, timeout: int = 30) -> requests.Response | None:
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True)
        if r.status_code != 200:
            return None
        return r
    except Exception:
        return None


def parse_sitemap_index(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    out = []
    for child in root:
        if strip_ns(child.tag) != "sitemap":
            continue
        for node in child:
            if strip_ns(node.tag) == "loc" and node.text:
                out.append(to_https(node.text.strip()))
    return out


def parse_urlset(xml_text: str) -> list[str]:
    root = ET.fromstring(xml_text)
    urls = []
    for child in root:
        if strip_ns(child.tag) != "url":
            continue
        for node in child:
            if strip_ns(node.tag) == "loc" and node.text:
                urls.append(to_https(node.text.strip()))
    return urls


def normalize_link(base_url: str, href: str) -> str | None:
    if not href:
        return None
    u = urljoin(base_url, href)
    u, _ = urldefrag(u)
    return u


def find_pdf_links_in_html(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    out = set()

    # Normal anchors
    for a in soup.select("a[href]"):
        u = normalize_link(base_url, a.get("href"))
        if u and PDF_RE.search(u):
            out.add(u)

    # Sometimes PDFs appear in embeds/scripts; do a quick regex scan too
    # (still normalize relative URLs if possible)
    for m in re.findall(r"""["']([^"']+\.pdf(?:\?[^"']*)?)["']""", html, flags=re.IGNORECASE):
        u = normalize_link(base_url, m)
        if u and PDF_RE.search(u):
            out.add(u)

    return out


def extract_pdf_text_bytes(pdf_bytes: bytes) -> str:
    if HAVE_PDFPLUMBER:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                parts = []
                for i, page in enumerate(pdf.pages):
                    txt = page.extract_text() or ""
                    txt = txt.strip()
                    if txt:
                        parts.append(f"[PAGE {i+1}]\n{txt}")
                return "\n\n".join(parts).strip()
        except Exception:
            # fallback to pypdf
            pass

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for i, page in enumerate(reader.pages):
            txt = page.extract_text() or ""
            txt = txt.strip()
            if txt:
                parts.append(f"[PAGE {i+1}]\n{txt}")
        return "\n\n".join(parts).strip()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sitemap_index", default="https://cis.haifa.ac.il/sitemap_index.xml")
    ap.add_argument(
        "--include_sitemaps",
        nargs="*",
        default=["page-sitemap.xml", "post-sitemap.xml", "staff-sitemap.xml"],
        help="Which sitemap files to scan for pages (substrings).",
    )
    ap.add_argument("--max_pages", type=int, default=2000)
    ap.add_argument("--max_pdfs", type=int, default=500)
    ap.add_argument("--delay", type=float, default=0.6)
    ap.add_argument("--out", default="cis_pdfs.jsonl")
    ap.add_argument("--domain", default="cis.haifa.ac.il")
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "UH-CIS-RAG-Bot/1.0 (academic project)",
        "Accept-Language": "he,en;q=0.9",
    })

    # 1) Fetch sitemap index
    idx_resp = fetch(session, args.sitemap_index)
    if not idx_resp:
        raise SystemExit(f"Failed to fetch sitemap index: {args.sitemap_index}")

    sitemap_urls = parse_sitemap_index(idx_resp.text)

    # 2) Choose sitemaps
    chosen = [sm for sm in sitemap_urls if any(p in sm for p in args.include_sitemaps)]
    if not chosen:
        raise SystemExit("No sitemaps matched. Try --include_sitemaps page-sitemap.xml post-sitemap.xml staff-sitemap.xml")

    print("Chosen sitemaps:")
    for sm in chosen:
        print(" -", sm)

    # 3) Collect page URLs from chosen sitemaps
    page_urls = []
    for sm in chosen:
        sm_resp = fetch(session, sm)
        if not sm_resp:
            continue
        page_urls.extend(parse_urlset(sm_resp.text))

    # Deduplicate + limit
    page_urls = list(dict.fromkeys(page_urls))[: args.max_pages]
    print(f"Total HTML pages to scan: {len(page_urls)}")

    # 4) Scan pages for PDF links
    pdf_to_sources = defaultdict(set)

    for url in tqdm(page_urls, desc="Scanning pages for PDFs"):
        r = fetch(session, url, timeout=25)
        if not r:
            continue

        ctype = (r.headers.get("Content-Type") or "").lower()
        if "text/html" not in ctype:
            continue

        pdfs = find_pdf_links_in_html(r.text, url)
        for p in pdfs:
            # Keep only PDFs hosted on the domain (optional, but matches your ask)
            if urlparse(p).netloc.endswith(args.domain):
                pdf_to_sources[p].add(url)

        time.sleep(args.delay)

        if len(pdf_to_sources) >= args.max_pdfs:
            break

    pdf_urls = list(pdf_to_sources.keys())[: args.max_pdfs]
    print(f"Found {len(pdf_urls)} unique PDF URLs (domain-filtered).")

    # 5) Download PDFs and extract text
    saved = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for pdf_url in tqdm(pdf_urls, desc="Downloading + extracting PDFs"):
            r = fetch(session, pdf_url, timeout=60)
            if not r:
                continue

            ctype = (r.headers.get("Content-Type") or "").lower()
            # Some servers return octet-stream; accept if URL is .pdf
            if ("pdf" not in ctype) and ("octet-stream" not in ctype) and (not PDF_RE.search(pdf_url)):
                continue

            text = extract_pdf_text_bytes(r.content)

            # If empty, it's likely scanned or extraction failed
            if len(text.strip()) < 200:
                # Still save a record so you know it exists (optional)
                record = {
                    "url": pdf_url,
                    "kind": "pdf",
                    "text": "",
                    "note": "No extractable text (possibly scanned PDF or encoding issue).",
                    "source_pages": sorted(list(pdf_to_sources[pdf_url]))[:20],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                time.sleep(args.delay)
                continue

            record = {
                "url": pdf_url,
                "kind": "pdf",
                "text": text,
                "source_pages": sorted(list(pdf_to_sources[pdf_url]))[:20],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            saved += 1
            time.sleep(args.delay)

    print(f"Done. PDFs saved (with extracted text when available) to: {args.out}")
    print(f"Extracted text successfully from {saved} PDFs.")


if __name__ == "__main__":
    main()
