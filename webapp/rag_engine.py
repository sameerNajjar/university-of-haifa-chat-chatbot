import json
import os
import re
import time
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# HybridRetriever import (support both layouts)
try:
    from chatbot.hybrid_retriever import HybridRetriever
except Exception:
    from hybrid_retriever import HybridRetriever


# Forbidden scripts: Cyrillic, Arabic, Hangul, CJK (Chinese/Japanese), etc.
FORBIDDEN_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF]"  # Cyrillic
    r"|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"  # Arabic + ext
    r"|[\uAC00-\uD7AF]"  # Hangul
    r"|[\u4E00-\u9FFF]"  # CJK Unified Ideographs
    r"|[\u3040-\u30FF]"  # Hiragana/Katakana
)

NUM_RE = re.compile(r"\d")


def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas


def hebrew_ratio(text: str) -> float:
    heb = 0
    alpha = 0
    for ch in text:
        if ch.isalpha():
            alpha += 1
            if "\u0590" <= ch <= "\u05FF":
                heb += 1
    return heb / max(1, alpha)


def is_hebrew(text: str) -> bool:
    return hebrew_ratio(text) >= 0.15


def needs_exact_number(q: str) -> bool:
    triggers = [
        # Hebrew
        "שכר לימוד", "כמה עולה", "עלות", "מחיר",
        "דדליין", "מועד אחרון", "תאריך", "מתי", "עד מתי",
        "מדויק", "בדיוק", "סכום",
        # English
        "tuition", "how much", "cost", "price",
        "deadline", "due date", "date", "when", "until when",
        "exact", "precise", "amount",
    ]
    q_norm = q.strip().lower()
    return any(t in q_norm for t in triggers)


def sources_have_numbers(picked: List[Tuple[float, Dict[str, Any]]]) -> bool:
    for _, m in picked:
        txt = (m.get("text") or "")
        if NUM_RE.search(txt):
            return True
    return False


def clean_forbidden_scripts(text: str) -> str:
    # Remove characters from forbidden scripts (keep Hebrew/Latin/digits/punct)
    return FORBIDDEN_SCRIPT_RE.sub("", text)


def build_sources_block(
    picked: List[Tuple[float, Dict[str, Any]]],
    max_chars_each: int = 1600
) -> str:
    parts = []
    for j, (score, m) in enumerate(picked, start=1):
        url = m.get("url", "")
        title = (m.get("title") or "").strip()
        text = (m.get("text") or "").strip()

        if len(text) > max_chars_each:
            text = text[:max_chars_each].rsplit("\n", 1)[0].strip() + "\n..."

        title_part = f" | {title}" if title else ""
        parts.append(f"[SOURCE {j}] {url}{title_part}\n{text}\n")
    return "\n".join(parts).strip()


def fit_sources_to_context(
    picked: List[Tuple[float, Dict[str, Any]]],
    max_tokens: int = 4000
) -> List[Tuple[float, Dict[str, Any]]]:
    # rough estimate: 1 token ≈ 4 chars
    max_chars = max_tokens * 4
    total_chars = 0
    fitted: List[Tuple[float, Dict[str, Any]]] = []

    for score, meta in picked:
        text = meta.get("text", "") or ""
        url = meta.get("url", "") or ""
        title = meta.get("title", "") or ""

        header_chars = len(url) + len(title) + 50
        remaining = max_chars - total_chars - header_chars

        if remaining < 200:
            break

        if len(text) > remaining:
            truncated = text[:remaining].rsplit(".", 1)[0]
            if not truncated:
                truncated = text[:remaining].rsplit("\n", 1)[0]
            text = truncated.strip() + "..."

        fitted_meta = dict(meta)
        fitted_meta["text"] = text
        fitted.append((score, fitted_meta))
        total_chars += len(text) + header_chars

    return fitted


SOURCES_FOOTER_LINE = re.compile(r"(?im)^\s*\[\d+\]\s*[—-].*$")

def strip_sources_footer(ans: str) -> str:
    lines = ans.splitlines()

    # stop if model starts a sources section
    out = []
    for line in lines:
        if line.strip().lower() in {"מקורות:", "sources:"}:
            break
        if SOURCES_FOOTER_LINE.match(line):  # lines like: [1] — title
            continue
        out.append(line)

    # remove markdown bold if model still uses it
    return "\n".join(out).replace("**", "").strip()

class RagEngine:
    def __init__(
        self,
        emb_path: str,
        meta_path: str,
        embed_model_name: str = "intfloat/multilingual-e5-small",
        ollama_url: str = "http://localhost:11434",
        llm_model: str = "qwen3:8b",
        topk: int = 5,
        num_ctx: int = 8192,
        alpha: float = 0.6,
        max_sources_tokens: int = 4000,
        max_chars_each: int = 1600,
        history_turns: int = 8,
        enforce_exact_numbers: bool = True,
    ):
        if not os.path.exists(emb_path):
            raise RuntimeError(f"Embeddings not found: {emb_path}")
        if not os.path.exists(meta_path):
            raise RuntimeError(f"Metadata not found: {meta_path}")

        # memory-friendly load
        E = np.load(emb_path, mmap_mode="r")
        if E.dtype != np.float32:
            E = E.astype(np.float32)

        metas = load_meta(meta_path)
        if len(metas) != E.shape[0]:
            raise RuntimeError("Meta rows must match embeddings rows")

        self.E = E
        self.metas = metas

        self.embed_model = SentenceTransformer(embed_model_name)

        self.ollama_url = ollama_url.rstrip("/")
        self.llm_model = llm_model
        self.topk = topk
        self.num_ctx = num_ctx

        self.max_sources_tokens = max_sources_tokens
        self.max_chars_each = max_chars_each
        self.history_turns = history_turns
        self.enforce_exact_numbers = enforce_exact_numbers

        self.retriever = HybridRetriever(self.E, self.metas, self.embed_model, alpha=alpha)

    def ollama_chat(self, messages, temperature=0.1, top_p=0.9) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p, "num_ctx": self.num_ctx},
        }
        r = requests.post(self.ollama_url + "/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def answer(
        self,
        query: str,
        want_hebrew: Optional[bool] = None,
        max_per_url: int = 2,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        q = (query or "").strip()
        if not q:
            return ("שאלה ריקה." if (want_hebrew is True) else "Empty question."), []

        # auto language unless forced
        if want_hebrew is None:
            want_he = is_hebrew(q)
        else:
            want_he = bool(want_hebrew)

        picked = self.retriever.retrieve(q, topk=self.topk, max_per_url=max_per_url)

        # exact-number guard
        if self.enforce_exact_numbers and needs_exact_number(q) and picked and (not sources_have_numbers(picked)):
            msg_he = (
                "לא מצאתי במקורות שסורקו סכום/מספר/תאריך מדויק שתומך בתשובה. "
                "כדי לענות רשמית, צריך להוסיף לאינדוקס עמוד/מסמך רשמי שמכיל את הנתון."
            )
            msg_en = (
                "I couldn’t find an exact number/date in the indexed sources to support a reliable answer. "
                "To answer officially, the relevant official page/document must be indexed."
            )
            return (msg_he if want_he else msg_en), []

        # fit sources into context window + clean source text
        picked = fit_sources_to_context(picked, max_tokens=self.max_sources_tokens)
        cleaned_picked: List[Tuple[float, Dict[str, Any]]] = []
        for score, m in picked:
            m2 = dict(m)
            m2["text"] = clean_forbidden_scripts(m2.get("text", "") or "")
            cleaned_picked.append((score, m2))

        sources_block = build_sources_block(cleaned_picked, max_chars_each=self.max_chars_each)

        system_msg = (
            "You are a RAG assistant for the University of Haifa Faculty of Computer & Information Science.\n"
            "STRICT RULES:\n"
            "1) Use ONLY the provided SOURCES. Do not use outside knowledge.\n"
            "2) If the answer is not explicitly supported by the sources, say you couldn't find it in the indexed pages.\n"
            "3) Always add citations like [1], [2] that refer to SOURCE numbers.\n"
            "4) Be concise and factual. Prefer bullet points when helpful.\n"
            f"5) Respond in {'Hebrew' if want_he else 'English'}.\n"
            "6) If sources conflict, mention both views and cite each.\n"
            "7) For numerical data (prices, dates, deadlines), quote EXACTLY from sources and cite.\n"
            "8) CRITICAL: Use ONLY Hebrew or English in your response. Do NOT use Arabic, Russian, Chinese, Korean, etc.\n"
        )

        user_msg = (
            f"Question:\n{q}\n\n"
            f"SOURCES:\n{sources_block}\n\n"
            "Write an answer with citations. Remember: ONLY Hebrew or English."
        )

        messages = [{"role": "system", "content": system_msg}]
        if history:
            messages.extend(history[-self.history_turns:])
        messages.append({"role": "user", "content": user_msg})

        start = time.time()
        ans = self.ollama_chat(messages, temperature=0.1, top_p=0.9)
        ans = strip_sources_footer(ans)
        _ = time.time() - start  # (kept if you want to log later)

        # If forbidden scripts appear: regenerate once, then force-clean
        if FORBIDDEN_SCRIPT_RE.search(ans):
            messages.append({
                "role": "user",
                "content": "Your previous response contained languages other than Hebrew/English. "
                           "Please answer again using ONLY Hebrew or English characters. "
                           "Keep citations like [1]. Do not add new facts."
            })
            ans2 = self.ollama_chat(messages, temperature=0.1, top_p=0.9)
            ans = ans2

        if FORBIDDEN_SCRIPT_RE.search(ans):
            ans = clean_forbidden_scripts(ans)

        # Return compact sources for UI
        src_list = []
        for j, (score, m) in enumerate(cleaned_picked, start=1):
            src_list.append({
                "n": j,
                "score": float(score),
                "url": m.get("url"),
                "title": m.get("title"),
            })

        return ans.strip(), src_list