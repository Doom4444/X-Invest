// static/js/chat.js
// Frontend for the X‑Invest chat page.
// Streaming via POST /api/chat/stream (NDJSON).
// Bubble style matches Ollama_Chatbot_UI:
//   - anonymous wrapper div (no class) so bubbles size to content
//   - escapeHtml + innerHTML for safe cursor injection
// Session ID persists in localStorage across page refreshes.

const sessionId =
  window.localStorage.getItem("xinvest_session") ||
  (window.crypto.randomUUID && window.crypto.randomUUID()) ||
  String(Date.now());
window.localStorage.setItem("xinvest_session", sessionId);

const messagesEl   = document.getElementById("messages");
const emptyStateEl = document.getElementById("emptyState");
const userInputEl  = document.getElementById("userInput");
const sendBtnEl    = document.getElementById("sendBtn");
const newChatBtnEl = document.getElementById("newChatBtn");
const statusEl     = document.getElementById("status");

let isSending = false;

// ── Helpers ───────────────────────────────────────────────────────────────────

// Converts plain text to safe HTML — lets us use innerHTML without XSS risk.
function escapeHtml(s) {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}

function setStatus(text, isError = false) {
  if (!statusEl) return;
  statusEl.textContent = text || "";
  statusEl.className = "hint" + (isError ? " error" : "");
}

function hideEmptyState() {
  if (emptyStateEl) emptyStateEl.style.display = "none";
}
function showEmptyState() {
  if (emptyStateEl) emptyStateEl.style.display = "flex";
}

function scrollToBottom() {
  window.scrollTo({ top: document.body.scrollHeight });
}

// ── Language detection ────────────────────────────────────────────────────────
// Prepended to the message sent to the backend (never shown in the UI).
// Reinforces the system prompt so ALLaM replies in the correct language.
function detectLanguageHint(text) {
  if (!text || text.length < 2) return "";
  if (/[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/.test(text))
    return "The user wrote in Arabic. You MUST respond entirely in Arabic.";
  if (/[\u4e00-\u9fff]/.test(text))
    return "The user wrote in Chinese. You MUST respond entirely in Chinese.";
  if (/[\u3040-\u309f\u30a0-\u30ff]/.test(text))
    return "The user wrote in Japanese. You MUST respond entirely in Japanese.";
  return "";
}

// ── Bubble rendering ──────────────────────────────────────────────────────────
// Single function that (re)writes the bubble content.
// showCursor=true appends the blinking cursor span while the model is streaming.
function setBubble(bubble, text, showCursor = false) {
  bubble.innerHTML =
    escapeHtml(text) +
    (showCursor ? '<span class="cursor"></span>' : "");
  scrollToBottom();
}

// ── Create a message row ──────────────────────────────────────────────────────
// DOM structure mirrors Ollama_Chatbot_UI exactly:
//
//   <div class="message user|assistant">
//     <span class="avatar">Y|X</span>
//     <div>                           ← anonymous wrapper, NO class
//       <div class="bubble">…</div>
//       <div class="meta"></div>
//     </div>
//   </div>
//
// The anonymous wrapper has no CSS applied to it, so it shrinks to the bubble's
// natural fit-content width. The CSS rule
//   .message.user > div:not(.avatar) { align-items: flex-end }
// pushes user bubbles to the right edge.
//
// Returns { bubble, meta } for the stream loop to write into.
function createMessageRow(role, content = "") {
  if (!messagesEl) return null;
  hideEmptyState();

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  // Avatar — use <span> to match old project
  const avatar = document.createElement("span");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "Y" : "X";

  // Anonymous wrapper div — NO className
  const body = document.createElement("div");
  body.className = "message-body";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = escapeHtml(content);  // empty for assistant; set for user

  const meta = document.createElement("div");
  meta.className = "meta";

  body.appendChild(bubble);
  body.appendChild(meta);
  wrapper.appendChild(avatar);
  wrapper.appendChild(body);
  messagesEl.appendChild(wrapper);

  scrollToBottom();
  return { bubble, meta };
}

// ── Main send ─────────────────────────────────────────────────────────────────
async function sendMessage() {
  if (!userInputEl || !sendBtnEl) return;
  const text = userInputEl.value.trim();
  if (!text || isSending) return;

  isSending = true;
  sendBtnEl.disabled = true;
  setStatus("Thinking…");

  // The language hint goes to the backend but the user sees only their raw text
  const hint          = detectLanguageHint(text);
  const messageToSend = hint ? `[${hint}]\n${text}` : text;

  createMessageRow("user", text);
  userInputEl.value = "";
  // Reset textarea height after clearing
  userInputEl.style.height = "auto";

  // Empty assistant bubble — filled token by token
  const assistant = createMessageRow("assistant", "");
  if (!assistant) {
    isSending = false;
    sendBtnEl.disabled = false;
    return;
  }
  const { bubble, meta } = assistant;

  // Show cursor immediately so the user knows the model is working
  setBubble(bubble, "", true);

  let fullText  = "";
  const startTime = Date.now();

  try {
    const res = await fetch("/api/chat/stream", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId, message: messageToSend }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    // ── NDJSON stream reader ──────────────────────────────────────────────────
    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Ollama sends one JSON object per newline
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";   // keep the incomplete last fragment

      for (const line of lines) {
        if (!line.trim()) continue;

        let data;
        try {
          data = JSON.parse(line);
        } catch {
          continue;   // malformed line — skip silently
        }

        // ── Backend error (e.g. Ollama unreachable) ───────────────────────
        if (data.error) {
          setBubble(bubble, "Sorry, something went wrong.\n" + data.error, false);
          setStatus(data.error, true);
          return;
        }

        // ── Sentinel from api/chat.py — stream fully done ─────────────────
        // Contains post-processed text (disclaimer guaranteed by postprocessor).
        // Replaces bubble content with the cleaned final version.
        if (data.x_invest_final) {
          setBubble(bubble, data.full_response || fullText, false);
          setStatus("");
          continue;
        }

        // ── Normal token chunk ────────────────────────────────────────────
        const token = data.message?.content ?? "";
        if (token) {
          fullText += token;
          setBubble(bubble, fullText, true);   // cursor visible while streaming
        }

        // ── Ollama done flag — show token / time stats ────────────────────
        if (data.done) {
          setBubble(bubble, fullText, false);
          const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
          const parts   = [];
          if (data.eval_count != null) parts.push(`Tokens: ${data.eval_count}`);
          parts.push(`Time: ${elapsed}s`);
          meta.textContent = parts.join(" · ");
          setStatus("");
        }
      }
    }

  } catch (e) {
    const msg = e?.message || "Network error";
    setBubble(bubble, "Sorry, something went wrong.\n" + msg, false);
    setStatus(msg, true);
  } finally {
    isSending = false;
    sendBtnEl.disabled = false;
  }
}

// ── New chat ──────────────────────────────────────────────────────────────────
async function newChat() {
  if (isSending) return;
  setStatus("Starting a fresh chat…");
  try {
    await fetch("/api/clear", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId }),
    });
  } catch {
    // ignore backend error — still clear the UI
  }
  if (messagesEl) {
    while (messagesEl.firstChild) messagesEl.removeChild(messagesEl.firstChild);
  }
  showEmptyState();
  setStatus("");
}

// ── Event listeners ───────────────────────────────────────────────────────────
if (userInputEl) {
  userInputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
  // Auto-grow textarea as user types
  userInputEl.addEventListener("input", () => {
    userInputEl.style.height = "auto";
    userInputEl.style.height = Math.min(userInputEl.scrollHeight, 12 * 24) + "px";
  });
}
if (sendBtnEl)    sendBtnEl.addEventListener("click", sendMessage);
if (newChatBtnEl) newChatBtnEl.addEventListener("click", newChat);
