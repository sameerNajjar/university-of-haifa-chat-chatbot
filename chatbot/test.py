# python ./chat_bot/rag_chat_bot.py --llm qwen3:8b --topk 5

import argparse
import json
import os
import sys
from typing import List, Dict, Any, Tuple
import re
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from hebrew_utils import detect_query_intent
from hybrid_retriever import HybridRetriever
from logger import ChatLogger
from language_filter import validate_response_language, clean_response, contains_unwanted_languages
import time

# Windows terminal Hebrew support
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NUM_RE = re.compile(r"\d")

def needs_exact_number(q: str) -> bool:
    triggers = [
        "שכר לימוד", "כמה עולה", "עלות", "מחיר",
        "דדליין", "מועד אחרון", "תאריך", "מתי", "עד מתי",
        "מדויק", "בדיוק", "סכום"
    ]
    q_norm = q.strip().lower()
    return any(t in q_norm for t in triggers)

def sources_have_numbers(picked) -> bool:
    # picked is list of (score, meta)
    for _, m in picked:
        txt = (m.get("text") or "")
        if NUM_RE.search(txt):
            return True
    return False

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


def retrieve_topk(
    query: str,
    E: np.ndarray,
    metas: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    topk: int = 6,
    max_per_url: int = 2,
) -> List[Tuple[float, Dict[str, Any]]]:
    # E5 format
    q = embed_model.encode(["query: " + query], normalize_embeddings=True)[0].astype(np.float32)

    sims = E @ q  # cosine similarity because normalized
    k = min(topk * 4, len(sims))  # overfetch then filter
    idxs = np.argpartition(-sims, k - 1)[:k]
    idxs = idxs[np.argsort(-sims[idxs])]

    picked: List[Tuple[float, Dict[str, Any]]] = []
    per_url = {}

    for i in idxs:
        m = metas[int(i)]
        url = m.get("url", "")
        if not url:
            continue
        per_url.setdefault(url, 0)
        if per_url[url] >= max_per_url:
            continue
        picked.append((float(sims[i]), m))
        per_url[url] += 1
        if len(picked) >= topk:
            break

    return picked


def build_sources_block(picked: List[Tuple[float, Dict[str, Any]]], max_chars_each: int = 1600) -> str:
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

def load_embeddings_mmap(path):
    """Memory-efficient loading for large embedding files"""
    return np.load(path, mmap_mode='r')

def ollama_chat(
    ollama_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    top_p: float = 0.9,
    num_ctx: int = 8192,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": num_ctx,
        },
    }
    r = requests.post(ollama_url.rstrip("/") + "/api/chat", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()["message"]["content"]

def validate_required_files(emb_path: str, meta_path: str) -> None:
    """Check that required index files exist before starting"""
    missing = []
    
    if not os.path.exists(emb_path):
        missing.append(f"Embeddings: {emb_path}")
    if not os.path.exists(meta_path):
        missing.append(f"Metadata: {meta_path}")
    
    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f"   - {m}")
        print("\nPlease run the pipeline first:")
        print("  1. python extract_tokens.py")
        print("  2. python clean_data.py")
        print("  3. python chunking.py")
        print("  4. python build_index.py")
        raise SystemExit(1)

def fit_sources_to_context(
    picked: List[Tuple[float, Dict[str, Any]]], 
    max_tokens: int = 4000
) -> List[Tuple[float, Dict[str, Any]]]:
    """
    Dynamically truncate sources to fit in context window
    
    Args:
        picked: Retrieved sources
        max_tokens: Rough token budget for sources
    
    Returns:
        Fitted sources with truncated text
    """
    # Rough estimate: 1 token ≈ 4 characters for mixed Hebrew/English
    max_chars = max_tokens * 4
    
    total_chars = 0
    fitted = []
    
    for score, meta in picked:
        text = meta.get('text', '')
        url = meta.get('url', '')
        title = meta.get('title', '')
        
        # Reserve space for URL and title
        header_chars = len(url) + len(title) + 50
        
        # Calculate remaining budget
        remaining = max_chars - total_chars - header_chars
        
        if remaining < 200:  # Not enough space
            print(f"  Context limit reached, keeping top {len(fitted)} sources")
            break
        
        # Truncate text if needed
        if len(text) > remaining:
            # Try to cut at sentence boundary
            truncated = text[:remaining].rsplit('.', 1)[0]
            if not truncated:  # No sentence boundary found
                truncated = text[:remaining].rsplit('\n', 1)[0]
            text = truncated.strip() + "..."
        
        # Create fitted metadata
        fitted_meta = {**meta, 'text': text}
        fitted.append((score, fitted_meta))
        total_chars += len(text) + header_chars
    
    print(f"  Using {len(fitted)}/{len(picked)} sources (~{total_chars} chars)")
    return fitted


def is_greeting(q: str) -> bool:
    q = q.strip().lower()
    # remove punctuation
    q = re.sub(r"[^\w\u0590-\u05FF]+", " ", q).strip()

    greetings = {
        "hi", "hello", "hey", "good morning", "good evening",
        "shalom", "salam",
        "שלום", "היי", "הי", "אהלן", "סלאם", "מה נשמע", "מה קורה"
    }

    # exact match or very short greeting phrases
    return q in greetings or any(q.startswith(g) for g in greetings)

def is_too_short(q: str) -> bool:
    q = q.strip()
    return len(q) < 4  # tweak if you want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="./data/cis_emb.npy")
    ap.add_argument("--meta", default="./data/cis_meta.jsonl")
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-small")
    ap.add_argument("--ollama_url", default="http://localhost:11434")
    ap.add_argument("--llm", default="qwen3:8b")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_per_url", type=int, default=2)
    ap.add_argument("--num_ctx", type=int, default=8192)
    args = ap.parse_args()

    validate_required_files(args.emb, args.meta)

    print("Loading embeddings + metadata...")
    E = np.load(args.emb).astype(np.float32)
    metas = load_meta(args.meta)
    if len(metas) != E.shape[0]:
        raise SystemExit(f"Meta rows ({len(metas)}) must match emb rows ({E.shape[0]}).")

    print(f"Loading embedding model: {args.embed_model}")
    
    try:
        embed_model = SentenceTransformer(args.embed_model)
    except Exception as e:
        raise SystemExit(f"Failed to load embedding model: {e}\n"
                        f"Try: pip install sentence-transformers --upgrade")
        
    history: List[Dict[str, str]] = []

    logger = ChatLogger("./data/chatbot_interactions.jsonl")
    print("Logging enabled to: chatbot_interactions.jsonl")

    print("Initializing hybrid retriever...")
    retriever = HybridRetriever(E, metas, embed_model, alpha=0.6)
    print("Hybrid retriever ready!")
    print("\nRAG Chat ready. Type 'exit' to quit.\n")

    while True:
        user_q = input("You: ").strip()
        if not user_q:
            continue
        if user_q.lower() in {"exit", "quit"}:
            break
        
        if user_q.lower() in {"reset", "clear"}:
            history.clear()
            print("\nAssistant:\nאיפסתי את השיחה. איך אפשר לעזור?\n")
            continue

        if is_greeting(user_q):
            history.clear()
            print("\nAssistant:\nשלום! איך אפשר לעזור? שאל אותי שאלה על הפקולטה/החוגים.\n")
            continue

        picked = retriever.retrieve(
            user_q, 
            topk=args.topk,
            max_per_url=args.max_per_url
        )
        
        if needs_exact_number(user_q) and not sources_have_numbers(picked):
            print("\nAssistant:\n"
          "לא מצאתי במקורות שסורקו סכום/מספר מדויק שמאפשר לענות על השאלה הזו. "
          "כדי לקבל נתון רשמי, צריך להוסיף לאינדוקס את עמוד/מסמך שכר הלימוד הרשמי של האוניברסיטה "
          "או מקור רשמי שמכיל את הסכום המדויק.\n")
            continue
            
        picked = fit_sources_to_context(picked, max_tokens=4000)
        sources = build_sources_block(picked)
        
        # Check for unwanted languages in sources
        for score, m in picked:
            text = m.get("text", "")
            has_unwanted, details = contains_unwanted_languages(text)
            if has_unwanted:
                print(f"⚠️  Warning: Source contains non-Hebrew/English text: {m.get('url')}")
                print(f"   Languages found: {details}")
                # Clean the source text
                m["text"] = clean_response(text)
        
        want_he = is_hebrew(user_q)

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
            "8) CRITICAL: Use ONLY Hebrew or English in your response. Do NOT use Arabic, Russian, Chinese, Korean, or any other languages.\n"
            "9) If you find yourself using other languages, stop and rewrite in Hebrew or English only.\n"
        )

        user_msg = (
            f"Question:\n{user_q}\n\n"
            f"SOURCES:\n{sources}\n\n"
            "Write an answer with citations. Remember: ONLY Hebrew or English!"
        )

        # Keep only short conversation history (no sources in history)
        messages = [{"role": "system", "content": system_msg}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_msg})

        try:
            start_time = time.time()
            ans = ollama_chat(
                ollama_url=args.ollama_url,
                model=args.llm,
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                num_ctx=args.num_ctx,
            )
            response_time = time.time() - start_time
            
            # Validate response language
            is_valid, error_msg = validate_response_language(ans, user_q)
            
            if not is_valid:
                print(f"\n⚠️  Warning: {error_msg}")
                print("   Cleaning response to remove unwanted languages...\n")
                ans = clean_response(ans)
                
                # If cleaned response is too short, regenerate
                if len(ans.strip()) < 50:
                    print("   Cleaned response too short. Asking LLM to regenerate...\n")
                    
                    # Add strict instruction to messages
                    messages.append({
                        "role": "user", 
                        "content": "Your previous response contained languages other than Hebrew/English. "
                                 "Please provide the answer again using ONLY Hebrew or English characters. "
                                 "Do not use Arabic (العربية), Russian (русский), or any other languages."
                    })
                    
                    ans = ollama_chat(
                        ollama_url=args.ollama_url,
                        model=args.llm,
                        messages=messages,
                        temperature=0.1,
                        top_p=0.9,
                        num_ctx=args.num_ctx,
                    )
                    
                    # Validate again
                    is_valid, error_msg = validate_response_language(ans, user_q)
                    if not is_valid:
                        print(f"   Still contains unwanted languages. Force-cleaning...")
                        ans = clean_response(ans)
            
            print("\nAssistant:\n" + ans.strip() + "\n")
            
            # Log the interaction
            query_intent = detect_query_intent(user_q)
            logger.log_interaction(
                query=user_q,
                answer=ans.strip(),
                sources=picked,
                query_lang='he' if want_he else 'en',
                response_time=response_time,
                metadata={
                    'query_intent': query_intent,
                    'language_valid': is_valid
                }
            )
            
        except requests.RequestException as e:
            print(f"\n[Error] Ollama call failed: {e}\n")
            print("Check: Ollama is running and model name is correct (try: ollama ls).")
            continue
        
        # Save minimal history (no giant context)
        history.append({"role": "user", "content": user_q})
        history.append({"role": "assistant", "content": ans.strip()})


if __name__ == "__main__":
    main()