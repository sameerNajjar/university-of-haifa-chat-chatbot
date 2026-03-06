// webapp/static/app.js
console.log("app.js loaded");

async function postForm(url, form) {
  const fd = new FormData(form);
  const r = await fetch(url, { method: "POST", body: fd });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.message || "Request failed");
  return data;
}

async function postJson(url, obj) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.message || "Request failed");
  return data;
}

window.addEventListener("DOMContentLoaded", () => {
  // login page
  const loginForm = document.getElementById("loginForm");
  const guestRoot = document.getElementById("guestRoot");
  if (guestRoot) initGuestUI();
  if (loginForm) {
    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const err = document.getElementById("msg") || document.getElementById("err");
      if (err) err.textContent = "";
      try {
        await postForm("/api/login", loginForm);
        window.location.href = "/chat";
      } catch (ex) {
        if (err) err.textContent = ex.message;
      }
    });
  }

  // register page
  const registerForm = document.getElementById("registerForm");
  if (registerForm) {
    registerForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const err = document.getElementById("msg") || document.getElementById("err");
      if (err) err.textContent = "";
      try {
        await postForm("/api/register", registerForm);
        window.location.href = "/login";
      } catch (ex) {
        if (err) err.textContent = ex.message;
      }
    });
  }

  // chat page
  const chatList = document.getElementById("chatList");
  if (chatList) initChatUI();
});

let currentChatId = null;

async function initChatUI() {
  const logout =
    document.getElementById("logoutBtn") || document.getElementById("logoutBtn2");
  if (logout) {
    logout.onclick = async () => {
      await fetch("/api/logout", { method: "POST" });
      window.location.href = "/login";
    };
  }

  const newChatBtn = document.getElementById("newChatBtn");
  if (newChatBtn) {
    newChatBtn.onclick = async () => {
      const c = await postJson("/api/chats", {});
      await loadChats(c.id);
    };
  }

  const sendBtn = document.getElementById("sendBtn");
  if (sendBtn) sendBtn.onclick = sendMessage;

  const inp = document.getElementById("q") || document.getElementById("msgInput");
  if (inp) {
    inp.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  await loadChats(null);
}

async function loadChats(selectId) {
  const chats = await fetch("/api/chats").then((r) => r.json());
  const chatList = document.getElementById("chatList");
  chatList.innerHTML = "";

  chats.forEach((c) => {
    const div = document.createElement("div");
    div.className = "chat-item";
    div.textContent = c.title;
    div.onclick = async () => {
      currentChatId = c.id;
      await loadMessages(c.id);
      highlightSelected(c.id);
    };
    div.dataset.id = c.id;
    chatList.appendChild(div);
  });

  if (chats.length === 0) {
    const c = await postJson("/api/chats", {});
    return loadChats(c.id);
  }

  const target = selectId || chats[0].id;
  currentChatId = target;
  await loadMessages(target);
  highlightSelected(target);
}

function highlightSelected(id) {
  document.querySelectorAll(".chat-item").forEach((el) => {
    el.classList.toggle("active", parseInt(el.dataset.id, 10) === id);
  });
}

async function loadMessages(chatId) {
  const msgs = await fetch(`/api/chats/${chatId}/messages`).then((r) => r.json());
  const box = document.getElementById("messages");
  box.innerHTML = "";
  msgs.forEach((m) => renderMsg(m.role, m.content, m.sources));
  box.scrollTop = box.scrollHeight;
}

function renderMsg(role, content, sources) {
  const box = document.getElementById("messages");
  const wrap = document.createElement("div");
  wrap.className = "msg " + (role === "user" ? "user" : "assistant");
  wrap.setAttribute("dir", "auto");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = content;
  bubble.setAttribute("dir", "auto");
  wrap.appendChild(bubble);

  if (role === "assistant" && sources && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.innerHTML = sources
      .map((s) => {
        const title = s.title ? ` — ${escapeHtml(s.title)}` : "";
        return `<div class="source"><a href="${escapeAttr(
          s.url
        )}" target="_blank">[${s.n}]</a>${title}</div>`;
      })
      .join("");
    wrap.appendChild(src);
  }

  box.appendChild(wrap);
}

async function sendMessage() {
  const inp = document.getElementById("q") || document.getElementById("msgInput");
  const text = (inp?.value || "").trim();
  if (!text) return;

  // If for any reason no chat is selected, create one.
  if (!currentChatId) {
    const c = await postJson("/api/chats", {});
    currentChatId = c.id;
    await loadChats(currentChatId);
  }

  inp.value = "";
  renderMsg("user", text);

  try {
    const res = await postJson(`/api/chats/${currentChatId}/send_async`, { text });
    renderMsg("assistant", res.answer, res.sources);
  } catch (e) {
    renderMsg("assistant", "Error: " + e.message);
  }

  const box = document.getElementById("messages");
  box.scrollTop = box.scrollHeight;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}
function escapeAttr(s) {
  return escapeHtml(s);
}


function initGuestUI() {
  const sendBtn = document.getElementById("guestSendBtn");
  const resetBtn = document.getElementById("guestResetBtn");
  const inp = document.getElementById("guestQ");
  const box = document.getElementById("messages");

  if (!sendBtn || !inp || !box) return;

  const doSend = async () => {
    const text = (inp.value || "").trim();
    if (!text) return;

    inp.value = "";
    renderMsg("user", text);

    try {
      const res = await postJson("/api/guest/send", { text });
      renderMsg("assistant", res.answer, res.sources);
    } catch (e) {
      renderMsg("assistant", "Error: " + e.message);
    }

    box.scrollTop = box.scrollHeight;
  };

  sendBtn.onclick = doSend;

  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      doSend();
    }
  });

  if (resetBtn) {
    resetBtn.onclick = () => {
      box.innerHTML =
        '<div class="msg assistant">Hi 👋 Ask me anything about the Faculty of Computer & Information Science (CS + IS).</div>';
      inp.value = "";
      inp.focus();
    };
  }
}