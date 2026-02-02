import json
import re
from typing import List, Dict, Any, Tuple
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from chatbot.hybrid_retriever import HybridRetriever



BAD_SCRIPT_RE = re.compile(
    r"[\u0400-\u04FF]"  # Cyrillic
    r"|[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"  # Arabic
    r"|[\uAC00-\uD7AF]"  # Hangul
)

def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    metas = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def build_sources_block(picked: List[Tuple[float, Dict[str, Any]]], max_chars_each: int = 1400) -> str:
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
    ):
        self.E = np.load(emb_path).astype(np.float32)
        self.metas = load_meta(meta_path)
        if len(self.metas) != self.E.shape[0]:
            raise RuntimeError("Meta rows must match embeddings rows")

        self.embed_model = SentenceTransformer(embed_model_name)

        self.ollama_url = ollama_url.rstrip("/")
        self.llm_model = llm_model
        self.topk = topk
        self.num_ctx = num_ctx

        # ✅ Use YOUR hybrid retriever
        self.retriever = HybridRetriever(self.E, self.metas, self.embed_model, alpha=alpha)

    def ollama_chat(self, messages, temperature=0.0, top_p=0.9) -> str:
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "top_p": top_p, "num_ctx": self.num_ctx},
        }
        r = requests.post(self.ollama_url + "/api/chat", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def answer(self, query: str, want_hebrew: bool = True, max_per_url: int = 2) -> Tuple[str, List[Dict[str, Any]]]:
        picked = self.retriever.retrieve(query, topk=self.topk, max_per_url=max_per_url)
        sources_block = build_sources_block(picked)

        system_msg = (
            "You are a RAG assistant for University of Haifa Faculty of Computer & Information Science.\n"
            "Rules:\n"
            "1) Use ONLY the SOURCES.\n"
            "2) If not found in sources, say so.\n"
            "3) Always cite with [1], [2] referring to SOURCE numbers.\n"
            "4) Output Hebrew (and if needed English) only. No other scripts.\n"
            f"5) Respond in {'Hebrew' if want_hebrew else 'English'}.\n"
        )
        user_msg = f"Question:\n{query}\n\nSOURCES:\n{sources_block}\n\nAnswer with citations."

        ans = self.ollama_chat(
            [{"role": "system", "content": system_msg},
             {"role": "user", "content": user_msg}],
            temperature=0.0,
        )

        # One rewrite pass if model outputs forbidden scripts
        if BAD_SCRIPT_RE.search(ans):
            rewrite_user = (
                "Rewrite the answer in Hebrew/English only (no Cyrillic/Arabic/Hangul). "
                "Keep the same meaning and keep citations like [1]. Do not add new facts.\n\n"
                f"ANSWER:\n{ans}"
            )
            ans = self.ollama_chat(
                [{"role": "system", "content": system_msg},
                 {"role": "user", "content": user_msg},
                 {"role": "user", "content": rewrite_user}],
                temperature=0.0,
            )

        # Return compact sources list for UI
        src_list = []
        for j, (score, m) in enumerate(picked, start=1):
            src_list.append({
                "n": j,
                "score": score,
                "url": m.get("url"),
                "title": m.get("title"),
            })

        return ans.strip(), src_list
