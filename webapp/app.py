from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import URLSafeSerializer
from sqlmodel import Session, select
from passlib.context import CryptContext

import json
import os

from .db import make_engine, init_db, User, Chat, Message
from .rag_engine import RagEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
DB_PATH = os.path.join(DATA_DIR, "app.db")

EMB_PATH = os.path.join(DATA_DIR, "cis_emb.npy")
META_PATH = os.path.join(DATA_DIR, "cis_meta.jsonl")

SECRET_KEY = os.environ.get("CHAT_SECRET_KEY", "dev-secret-change-me")
SESSION_SALT = "session"

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")
signer = URLSafeSerializer(SECRET_KEY, salt=SESSION_SALT)

engine = make_engine(DB_PATH)

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

os.makedirs(DATA_DIR, exist_ok=True)
init_db(engine)

rag = RagEngine(
    emb_path=EMB_PATH,
    meta_path=META_PATH,
    ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
    llm_model=os.environ.get("LLM_MODEL", "qwen3:8b"),
    topk=int(os.environ.get("TOPK", "5")),
    num_ctx=int(os.environ.get("NUM_CTX", "8192")),
    alpha=float(os.environ.get("ALPHA", "0.6")),
)


def get_db():
    with Session(engine) as s:
        yield s

def get_user_id_from_cookie(req: Request):
    token = req.cookies.get("session")
    if not token:
        return None
    try:
        data = signer.loads(token)
        return int(data["user_id"])
    except Exception:
        return None

def require_user(req: Request, db: Session = Depends(get_db)) -> User:
    uid = get_user_id_from_cookie(req)
    if not uid:
        raise HTTPException(status_code=401, detail="Not logged in")
    user = db.get(User, uid)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")
    return user

@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    uid = get_user_id_from_cookie(request)
    if uid:
        user = db.get(User, uid)
        if user:
            return RedirectResponse("/chat", status_code=303)
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/api/register")
def api_register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    username = username.strip()
    if len(username) < 3 or len(password) < 6:
        raise HTTPException(400, "Username>=3, password>=6")

    existing = db.exec(select(User).where(User.username == username)).first()
    if existing:
        raise HTTPException(400, "Username already exists")

    u = User(username=username, password_hash=pwd_context.hash(password))
    db.add(u)
    db.commit()
    db.refresh(u)
    return {"ok": True}

@app.post("/api/login")
def api_login(response: JSONResponse, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    u = db.exec(select(User).where(User.username == username.strip())).first()
    if not u or not pwd_context.verify(password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")

    token = signer.dumps({"user_id": u.id})
    resp = JSONResponse({"ok": True})
    resp.set_cookie("session", token, httponly=True, samesite="lax")
    return resp

@app.post("/api/logout")
def api_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("session")
    return resp

@app.get("/", response_class=HTMLResponse)
def chat_page(request: Request, user: User = Depends(require_user)):
    return templates.TemplateResponse("chat.html", {"request": request, "username": user.username})

@app.get("/api/chats")
def api_list_chats(user: User = Depends(require_user), db: Session = Depends(get_db)):
    chats = db.exec(select(Chat).where(Chat.user_id == user.id).order_by(Chat.id.desc())).all()
    return [{"id": c.id, "title": c.title, "created_at": c.created_at.isoformat()} for c in chats]

@app.post("/api/chats")
def api_new_chat(user: User = Depends(require_user), db: Session = Depends(get_db)):
    c = Chat(user_id=user.id, title="New chat")
    db.add(c)
    db.commit()
    db.refresh(c)
    return {"id": c.id, "title": c.title}

@app.get("/api/chats/{chat_id}/messages")
def api_chat_messages(chat_id: int, user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(404, "Chat not found")

    msgs = db.exec(select(Message).where(Message.chat_id == chat_id).order_by(Message.id.asc())).all()
    out = []
    for m in msgs:
        out.append({
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources_json) if m.sources_json else None,
            "created_at": m.created_at.isoformat(),
        })
    return out

@app.post("/api/chats/{chat_id}/send")
def api_send(chat_id: int, request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    body = request.json()  # FastAPI will await automatically? (not in sync) -> use async below
    raise HTTPException(500, "Use /api/chats/{chat_id}/send_async")

# Make it async to read json body properly
from fastapi import Body

@app.post("/api/chats/{chat_id}/send_async")
def api_send_async(chat_id: int, payload: dict = Body(...), user: User = Depends(require_user), db: Session = Depends(get_db)):
    chat = db.get(Chat, chat_id)
    if not chat or chat.user_id != user.id:
        raise HTTPException(404, "Chat not found")

    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "Empty message")

    # Store user message
    db.add(Message(chat_id=chat_id, role="user", content=text))
    db.commit()

    # Update chat title on first user message
    if chat.title == "New chat":
        chat.title = text[:40]
        db.add(chat)
        db.commit()

    # RAG answer
    want_he = True  # based on your audience; you can detect language if you want
    ans, sources = rag.answer(text, want_hebrew=want_he)

    # Store assistant message
    db.add(Message(chat_id=chat_id, role="assistant", content=ans, sources_json=json.dumps(sources, ensure_ascii=False)))
    db.commit()

    return {"answer": ans, "sources": sources}
