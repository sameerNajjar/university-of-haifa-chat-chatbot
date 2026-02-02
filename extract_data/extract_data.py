# python ./extract_data/extract_data.py --include page-sitemap.xml staff-sitemap.xml post-sitemap.xml https://cis.haifa.ac.il/category-sitemap.xml --out cis_pages.jsonl 

import argparse
import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
import trafilatura
from trafilatura.metadata import extract_metadata
from tqdm import tqdm


SITEMAP_INDEX = "https://cis.haifa.ac.il/sitemap_index.xml"


def strip_ns(tag: str) -> str:
    # "{namespace}loc" -> "loc"
    return tag.split("}", 1)[-1] if "}" in tag else tag


def to_https(url: str) -> str:
    # Yoast sometimes publishes http:// URLs; normalize to https://
    if url.startswith("http://"):
        return "https://" + url[len("http://"):]
    return url


def fetch_text(session, url, timeout=25, retries=3):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code != 200:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None
            return r.text
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"Failed to fetch {url}: {e}")
            return None
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


def parse_urlset(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    urls = []
    for child in root:
        if strip_ns(child.tag) != "url":
            continue
        rec = {"loc": None, "lastmod": None}
        for node in child:
            t = strip_ns(node.tag)
            if t == "loc" and node.text:
                rec["loc"] = to_https(node.text.strip())
            elif t == "lastmod" and node.text:
                rec["lastmod"] = node.text.strip()
        if rec["loc"]:
            urls.append(rec)
    return urls


def looks_useful_url(url: str) -> bool:
    """
    Skip common non-content pages.
    You can relax/tighten these rules based on what you want indexed.
    """
    bad_parts = ["/tag/", "/category/", "/author/", "/wp-json/", "/feed/"]
    return not any(p in url for p in bad_parts)


def hebrew_ratio(text: str) -> float:
    # Simple heuristic: fraction of Hebrew letters among alphabetic chars
    heb = 0
    alpha = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            if "\u0590" <= ch <= "\u05FF":
                heb += 1
    return heb / max(1, alpha)


def detect_lang(text: str) -> str:
    r = hebrew_ratio(text)
    if r >= 0.20:
        return "he"
    return "non-he"  # could be en/other


def extract_main_text(html: str, url: str) -> tuple[str, str]:
    # Main content extraction (Hebrew preserved)
    text = trafilatura.extract(
        html,
        url=url,
        include_tables=True,
        include_comments=False,
        favor_precision=True,
    ) or ""

    # Robust title extraction without trafilatura.metadata
    title = ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        t = soup.find("title")
        if t and t.get_text(strip=True):
            title = t.get_text(strip=True)
    except Exception:
        pass

    # Normalize whitespace a bit
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return title, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cis_pages.jsonl")
    ap.add_argument("--delay", type=float, default=0.8)
    ap.add_argument("--max-urls", type=int, default=5000)

    # Choose which Yoast sitemaps to include
    ap.add_argument(
        "--include",
        nargs="*",
        default=["page-sitemap.xml", "staff-sitemap.xml"],
        help="Substring filters for sitemap URLs (e.g., page-sitemap.xml staff-sitemap.xml post-sitemap.xml)",
    )
    args = ap.parse_args()

    session = requests.Session()
    session.headers.update({
        "User-Agent": "UH-CIS-RAG-Bot/1.0 (academic project)",
        "Accept-Language": "he,en;q=0.9",
    })

    # 1) Fetch sitemap index
    idx_xml = fetch_text(session, SITEMAP_INDEX)
    if not idx_xml:
        raise SystemExit(f"Failed to fetch: {SITEMAP_INDEX}")

    sitemap_urls = parse_sitemap_index(idx_xml)

    # 2) Filter sitemaps
    chosen_sitemaps = [sm for sm in sitemap_urls if any(pat in sm for pat in args.include)]
    if not chosen_sitemaps:
        raise SystemExit("No sitemaps matched. Try: --include page-sitemap.xml staff-sitemap.xml")

    print("Chosen sitemaps:")
    for sm in chosen_sitemaps:
        print(" -", sm)

    # 3) Collect URLs from those sitemaps
    all_entries = []
    for sm in chosen_sitemaps:
        sm_xml = fetch_text(session, sm)
        if not sm_xml:
            continue
        entries = parse_urlset(sm_xml)
        for e in entries:
            e["source_sitemap"] = sm
        all_entries.extend(entries)

    # Deduplicate
    dedup = {}
    for e in all_entries:
        dedup[e["loc"]] = e
    entries = list(dedup.values())

    # Keep only cis.haifa.ac.il domain + content-like URLs
    entries = [
        e for e in entries
        if urlparse(e["loc"]).netloc == "cis.haifa.ac.il" and looks_useful_url(e["loc"])
    ][: args.max_urls]

    print(f"Total URLs to fetch: {len(entries)}")

    # 4) Fetch each page + extract
    saved = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for e in tqdm(entries):
            url = e["loc"]
            html = fetch_text(session, url)
            if not html:
                continue

            title, text = extract_main_text(html, url)

            # Skip very short / empty pages
            if len(text) < 50:
                continue

            rec = {
                "url": url,
                "title": title,
                "text": text,                # UTF-8 Hebrew preserved
                "lang_guess": detect_lang(text),
                "hebrew_ratio": round(hebrew_ratio(text), 4),
                "lastmod": e.get("lastmod"),
                "source_sitemap": e.get("source_sitemap"),
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            saved += 1
            time.sleep(args.delay)

    print(f"Saved {saved} pages into: {args.out}")


if __name__ == "__main__":
    main()
