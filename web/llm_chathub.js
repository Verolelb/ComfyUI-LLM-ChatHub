/**
 * LLM ChatHub - chat panel for the ComfyUI sidebar.
 *
 * Talks to the /llm_chathub API exposed by this node pack: streaming replies,
 * persistent history, message editing/regeneration, image attachments and
 * document RAG. The same conversations are shared with the graph nodes.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const BASE = "/llm_chathub";
const USE_PANEL = "panel (saved settings)";

const PROVIDERS = [
  { value: "lmstudio", label: "LM Studio (local)" },
  { value: "ollama", label: "Ollama (local)" },
  { value: "ollama_cloud", label: "Ollama Cloud" },
];

/** Extensions accepted by the RAG import (server side: rag.TEXT_EXTENSIONS). */
const TEXT_ACCEPT = ".txt,.md,.markdown,.rst,.log,.csv,.tsv,.json,.yaml,.yml,.ini,.toml,.cfg,.conf,.env,.py,.js,.mjs,.ts,.html,.css,.xml,.srt,.vtt,.sql,.sh,.bat,.ps1,.c,.h,.cpp,.cs,.java,.go,.rs,.lua,.php,.rb,.kt,.swift,.prompt,.caption";

const state = {
  settings: null,
  conversations: [],
  conversation: null,
  messages: [],
  pending: [],
  draft: "",
  streaming: false,
  streamingId: null,
  controller: null,
  showPanel: false,
  index: { documents: 0, chunks: 0, embedded: 0, folder: "" },
  documents: [],
  modelCache: {},
  composerHeight: 0,
  status: "",
  error: "",
};

let rootEl = null;

/* ------------------------------------------------------------------ helpers */

function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value !== null && value !== undefined && value !== false) {
      node.setAttribute(key, value === true ? "" : value);
    }
  }
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

function escapeHtml(text) {
  return String(text ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function formatSize(bytes) {
  const value = Number(bytes) || 0;
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function readFileBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(new Error(`Could not read: ${file.name}`));
    reader.readAsDataURL(file);
  });
}

function renderMarkdown(text) {
  const blocks = [];
  let source = String(text ?? "");
  source = source.replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_match, _lang, code) => {
    blocks.push(code);
    return `\u0000CODE${blocks.length - 1}\u0000`;
  });
  let html = escapeHtml(source);
  html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
  html = html.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g,
    '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
  html = html.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(^|\n)#{1,4}\s*([^\n]+)/g, "$1<strong>$2</strong>");
  html = html.replace(/(^|\n)\s*[-*]\s+([^\n]+)/g, "$1&bull; $2");
  html = html.replace(/\n/g, "<br>");
  html = html.replace(/\u0000CODE(\d+)\u0000/g,
    (_match, index) => `<pre><code>${escapeHtml(blocks[Number(index)])}</code></pre>`);
  return html;
}

async function getJSON(path) {
  const response = await fetch(api.apiURL(path));
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

async function postJSON(path, body) {
  const response = await fetch(api.apiURL(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

/** POST + read a `text/event-stream` response, falling back to plain JSON. */
async function streamRequest(path, body, handlers = {}, signal = null) {
  const response = await fetch(api.apiURL(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
    signal,
  });

  if (!response.ok) {
    const raw = await response.text().catch(() => "");
    let message = `HTTP ${response.status}`;
    try {
      const parsed = JSON.parse(raw);
      message = parsed.error || message;
    } catch { /* keep the status */ }
    throw new Error(message);
  }

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("text/event-stream")) {
    const data = await response.json().catch(() => ({}));
    if (data.error) throw new Error(data.error);
    handlers.onResult?.(data);
    return data;
  }

  function dispatch(chunk) {
    let event = "message";
    let payload = "";
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) payload += line.slice(5).trim();
    }
    if (!payload) return;
    let data = null;
    try { data = JSON.parse(payload); } catch { return; }
    if (event === "delta") handlers.onDelta?.(data);
    else if (event === "reasoning") handlers.onReasoning?.(data);
    else if (event === "progress") handlers.onProgress?.(data);
    else if (event === "result") handlers.onResult?.(data);
    else if (event === "error") handlers.onError?.(data);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? "";
    for (const part of parts) if (part.trim()) dispatch(part);
  }
  if (buffer.trim()) dispatch(buffer);
  handlers.onEnd?.();
  return null;
}

/* ------------------------------------------------------------------ css */

const CSS = `
.llmch-root { display:flex; flex-direction:column; height:100%; width:100%; gap:6px;
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif; font-size:12px;
  color:var(--fg-color,#e6e6e6); background:var(--comfy-menu-bg,#1e1e1e);
  box-sizing:border-box; padding:6px; padding-right:12px; overflow:hidden;
  position:relative; }
/* Grip on the right edge: ComfyUI's own splitter handle is one pixel wide and
   invisible, so the panel looked fixed-width. The strip covers the whole edge
   (top to bottom, over the transcript and the composer) and draws a visible seam
   plus a knob, so it reads as a handle instead of a scrollbar. */
.llmch-grip { position:absolute; top:0; right:0; bottom:0; width:10px; z-index:6;
  cursor:col-resize; touch-action:none; }
.llmch-grip:before { content:""; position:absolute; top:0; bottom:0; right:3px; width:2px;
  border-radius:2px; background:var(--border-color,#3a3a3a); opacity:.75;
  transition:opacity .12s ease, background .12s ease; }
.llmch-grip:after { content:""; position:absolute; right:0; top:50%; width:7px; height:64px;
  transform:translateY(-50%); border-radius:4px; background:var(--border-color,#4a4a4a);
  border:1px solid var(--border-color,#5a5a5a); box-shadow:0 0 0 1px rgba(0,0,0,.25);
  transition:background .12s ease, height .12s ease; }
/* Horizontal grip: the top edge of the message box, so the input can be made
   taller by dragging it up (the browser only offers its own grip at the bottom). */
.llmch-grip-h { position:relative; height:9px; margin:-1px 0 0; cursor:row-resize;
  touch-action:none; display:flex; align-items:center; justify-content:center; }
.llmch-grip-h:before { content:""; position:absolute; left:0; right:0; top:50%; height:2px;
  transform:translateY(-50%); border-radius:2px; background:var(--border-color,#3a3a3a);
  opacity:.6; transition:opacity .12s ease, background .12s ease; }
.llmch-grip-h:after { content:""; position:relative; width:38px; height:5px; border-radius:3px;
  background:var(--border-color,#4a4a4a); border:1px solid var(--border-color,#5a5a5a);
  transition:background .12s ease; }
.llmch-grip-h:hover:before, .llmch-grip-h.llmch-dragging:before,
.llmch-grip-h:hover:after, .llmch-grip-h.llmch-dragging:after {
  background:var(--p-primary-color,#3b82f6); opacity:1; }
.llmch-grip-badge { position:absolute; font-size:10px; padding:2px 6px; border-radius:6px;
  white-space:nowrap; color:#fff; background:var(--p-primary-color,#3b82f6); pointer-events:none; }
.llmch-grip:hover:before, .llmch-grip.llmch-dragging:before,
.llmch-grip:hover:after, .llmch-grip.llmch-dragging:after {
  background:var(--p-primary-color,#3b82f6); opacity:1; }
.llmch-grip:hover:after, .llmch-grip.llmch-dragging:after { height:88px; }
.llmch-root * { box-sizing:border-box; }
.llmch-row { display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.llmch-grow { flex:1 1 auto; min-width:0; }
.llmch-btn { border:1px solid var(--border-color,#3a3a3a); background:var(--comfy-input-bg,#2b2b2b);
  color:inherit; border-radius:6px; padding:4px 8px; cursor:pointer; font-size:11px; line-height:1.4; }
.llmch-btn:hover { background:var(--comfy-menu-bg-hover,#3a3a3a); }
.llmch-btn.primary { background:var(--p-primary-color,#3b82f6); border-color:transparent; color:#fff; font-weight:600; }
.llmch-btn.danger { border-color:#a33; color:#f88; }
.llmch-btn:disabled { opacity:.45; cursor:not-allowed; }
.llmch-input, .llmch-textarea, .llmch-select { width:100%; background:var(--comfy-input-bg,#262626);
  color:inherit; border:1px solid var(--border-color,#3a3a3a); border-radius:6px; padding:5px 7px;
  font:inherit; outline:none; }
.llmch-textarea { resize:vertical; min-height:52px; line-height:1.45; }
.llmch-input:focus, .llmch-textarea:focus, .llmch-select:focus { border-color:var(--p-primary-color,#3b82f6); }
.llmch-chip { display:inline-flex; gap:4px; align-items:center; border:1px solid var(--border-color,#3a3a3a);
  border-radius:999px; padding:2px 8px; font-size:10px; opacity:.85; background:transparent; color:inherit; cursor:pointer; }
.llmch-chip.on { border-color:var(--p-primary-color,#3b82f6); color:var(--p-primary-color,#8ab4f8); opacity:1; }
.llmch-panel { border:1px solid var(--border-color,#3a3a3a); border-radius:8px; padding:8px;
  display:flex; flex-direction:column; gap:8px; overflow-y:auto; max-height:52%; background:var(--comfy-menu-bg,#202020); }
/* While the settings are open they own the height: the transcript and the status chips
   step aside so every option is on screen instead of inside a small scrolling box. */
.llmch-root.settings-open .llmch-panel { flex:1 1 auto; min-height:0; max-height:none; gap:4px; padding:6px; }
.llmch-root.settings-open .llmch-messages,
.llmch-root.settings-open .llmch-composer,
.llmch-root.settings-open .llmch-chips { display:none; }
.llmch-root.settings-open .llmch-section { gap:3px; }
.llmch-root.settings-open .llmch-input,
.llmch-root.settings-open .llmch-select { padding:3px 6px; }
.llmch-root.settings-open .llmch-btn { padding:3px 7px; }
.llmch-root.settings-open .llmch-files { max-height:64px; }
.llmch-section { display:flex; flex-direction:column; gap:5px; }
.llmch-label { font-size:10px; text-transform:uppercase; letter-spacing:.06em; opacity:.6; }
.llmch-messages { flex:1 1 auto; overflow-y:auto; display:flex; flex-direction:column; gap:8px;
  padding:6px 2px; min-height:80px; }
.llmch-msg { border:1px solid var(--border-color,#333); border-radius:8px; padding:7px 9px;
  background:var(--comfy-input-bg,#242424); }
.llmch-msg.user { border-color:#3b5f8a; background:#1d2a3a; }
.llmch-msg.assistant { border-color:#3d6b46; background:#1c2a1f; }
.llmch-msg.error { border-color:#a33; background:#331d1d; }
.llmch-msg-head { display:flex; gap:6px; align-items:center; margin-bottom:4px; font-size:10px; opacity:.65; }
.llmch-msg-body { line-height:1.5; white-space:normal; word-wrap:break-word; overflow-wrap:anywhere; }
.llmch-msg-body pre { background:#111; border-radius:6px; padding:6px; overflow-x:auto; margin:4px 0; }
.llmch-msg-body code { background:#111; padding:1px 4px; border-radius:4px; font-size:11px; }
.llmch-msg-actions { display:flex; gap:4px; margin-top:6px; flex-wrap:wrap; opacity:.75; }
.llmch-thumbs { display:flex; gap:4px; flex-wrap:wrap; margin:4px 0; }
.llmch-thumbs img { height:56px; border-radius:6px; border:1px solid var(--border-color,#444); cursor:zoom-in; }
.llmch-reasoning { border:1px dashed var(--border-color,#444); border-radius:6px; padding:4px 6px;
  margin:4px 0; font-size:11px; opacity:.85; }
.llmch-reasoning summary { cursor:pointer; opacity:.8; }
.llmch-reasoning .llmch-think { white-space:pre-wrap; overflow-wrap:anywhere; max-height:220px;
  overflow-y:auto; margin-top:4px; opacity:.85; }
.llmch-note { font-style:italic; opacity:.75; }
.llmch-files { display:flex; flex-direction:column; gap:3px; max-height:110px; overflow-y:auto; }
.llmch-file { display:flex; gap:6px; align-items:center; border:1px solid var(--border-color,#333);
  border-radius:6px; padding:2px 6px; font-size:11px; }
.llmch-file-name { flex:1 1 auto; min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.llmch-file-size { opacity:.6; font-size:10px; white-space:nowrap; }
.llmch-status { font-size:10px; opacity:.7; min-height:14px; white-space:pre-wrap; }
.llmch-status.error { color:#f88; opacity:1; }
.llmch-empty { opacity:.5; text-align:center; padding:14px 8px; }
.llmch-badge { font-size:10px; opacity:.7; border:1px dashed var(--border-color,#444); border-radius:6px; padding:2px 6px; }
.llmch-caret:after { content:"\\258B"; animation:llmch-blink 1s steps(2) infinite; margin-left:2px; }
@keyframes llmch-blink { 50% { opacity:0; } }
.llmch-overlay { position:fixed; right:16px; bottom:16px; width:420px; height:640px; z-index:9999;
  border:1px solid var(--border-color,#444); border-radius:10px; box-shadow:0 10px 40px rgba(0,0,0,.5);
  overflow:hidden; background:var(--comfy-menu-bg,#1e1e1e); }
.llmch-modal { position:fixed; inset:0; z-index:10000; background:rgba(0,0,0,.55);
  display:flex; align-items:center; justify-content:center; }
.llmch-modal-body { width:min(560px,92vw); max-height:82vh; overflow:auto; padding:14px; border-radius:10px;
  border:1px solid var(--border-color,#444); box-shadow:0 10px 40px rgba(0,0,0,.6);
  background:var(--comfy-menu-bg,#1e1e1e); color:var(--fg-color,#e6e6e6);
  display:flex; flex-direction:column; gap:8px; font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  font-size:12px; box-sizing:border-box; }
.llmch-modal-body * { box-sizing:border-box; }
`;

function injectCss() {
  if (document.getElementById("llmch-style")) return;
  document.head.append(el("style", { id: "llmch-style", text: CSS }));
}

/* ------------------------------------------------------------------ panel width */

const SIDE_WIDTH_KEY = "llmch.sidebar.width";
const SIDE_MIN_WIDTH = 280;
const SIDE_MIN_CANVAS = 160;

const COMPOSER_HEIGHT_KEY = "llmch.composer.height";
const COMPOSER_MIN_HEIGHT = 64;

/**
 * The ComfyUI panel that hosts the tab (absent in the floating fallback).
 *
 * Current builds put sidebar tabs in `.side-bar-panel`; older ones and the right
 * panel use other wrappers, so fall back to "the ancestor that sits next to a
 * splitter handle", which is exactly the panel we are allowed to resize.
 */
function sidebarPanel() {
  if (!rootEl) return null;
  try {
    const known = rootEl.closest(".side-bar-panel, #first-side-panel, #right-side-panel, .right-side-panel");
    if (known) return known;
    let node = rootEl;
    while (node && node.parentElement && node.parentElement !== document.body) {
      const parent = node.parentElement;
      const resizable = [...parent.children].some(
        (child) => child !== node && child.getAttribute("role") === "separator",
      );
      if (resizable) return node;
      node = parent;
    }
  } catch { /* detached node */ }
  return null;
}

/**
 * Resize the sidebar through the same flex percentages ComfyUI's splitter uses,
 * so the neighbouring panel keeps the rest of the row instead of overflowing.
 */
function applySidebarWidth(width, persist = true) {
  const panel = sidebarPanel();
  const group = panel?.parentElement;
  if (!panel || !group) return 0;
  const others = [...group.children].filter(
    (child) => child !== panel && child.getAttribute("role") !== "separator",
  );
  const total = group.getBoundingClientRect().width || 0;
  if (!total) return 0;
  const largest = Math.max(SIDE_MIN_WIDTH, total - SIDE_MIN_CANVAS);
  const clamped = Math.round(Math.min(Math.max(Number(width) || 0, SIDE_MIN_WIDTH), largest));
  const percent = (clamped / total) * 100;
  panel.style.flex = `${percent} 1 0px`;
  const rest = Math.max(0, 100 - percent) / Math.max(1, others.length);
  for (const other of others) other.style.flex = `${rest} 1 0px`;
  if (persist) {
    try { localStorage.setItem(SIDE_WIDTH_KEY, String(clamped)); } catch { /* private mode */ }
  }
  return clamped;
}

function savedSidebarWidth() {
  try {
    const value = Number(localStorage.getItem(SIDE_WIDTH_KEY));
    return Number.isFinite(value) && value > 0 ? value : 0;
  } catch {
    return 0;
  }
}

function restoreSidebarWidth() {
  const width = savedSidebarWidth();
  if (width) applySidebarWidth(width, false);
}

/** Height of the message box, remembered between sessions (0 = never dragged). */
function savedComposerHeight() {
  try {
    const value = Number(localStorage.getItem(COMPOSER_HEIGHT_KEY));
    return Number.isFinite(value) && value > 0 ? value : 0;
  } catch {
    return 0;
  }
}

function applyComposerHeight(textarea, height) {
  const value = Math.max(COMPOSER_MIN_HEIGHT, Math.round(Number(height) || 0));
  state.composerHeight = value;
  if (textarea) textarea.style.height = `${value}px`;
  return value;
}

/**
 * Grip on the top edge of the message box.
 *
 * The browser only gives a `<textarea>` a grip in its bottom-right corner, so the
 * input could only be made taller by pulling that corner down. This one sits on the
 * top edge: pull it up and the box grows instead of shrinking the transcript.
 */
function composerResizeGrip(textarea) {
  const grip = el("div", {
    class: "llmch-grip-h",
    title: "Drag up or down to resize the message box",
  });
  grip.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const startY = event.clientY;
    const startHeight = textarea.getBoundingClientRect().height;
    const room = rootEl ? rootEl.getBoundingClientRect().height : 600;
    const largest = Math.max(COMPOSER_MIN_HEIGHT + 40, Math.round(room * 0.55));
    grip.classList.add("llmch-dragging");
    try { grip.setPointerCapture(event.pointerId); } catch { /* synthetic pointer */ }
    const badge = el("span", {
      class: "llmch-grip-badge",
      style: "left:50%;bottom:12px;transform:translateX(-50%)",
      text: `${Math.round(startHeight)} px`,
    });
    grip.append(badge);

    const onMove = (moveEvent) => {
      // Upwards grows the box: the delta is inverted on purpose.
      const wanted = startHeight + (startY - moveEvent.clientY);
      const next = Math.min(Math.max(wanted, COMPOSER_MIN_HEIGHT), largest);
      applyComposerHeight(textarea, next);
      badge.textContent = `${state.composerHeight} px`;
      moveEvent.preventDefault();
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      grip.classList.remove("llmch-dragging");
      badge.remove();
      try { localStorage.setItem(COMPOSER_HEIGHT_KEY, String(state.composerHeight)); } catch { /* private mode */ }
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });
  return grip;
}

/** Drag handle on the right edge of the panel: resizes the whole left sidebar. */
function resizeGrip() {
  if (!sidebarPanel()) return null; // floating fallback: nothing to resize
  const grip = el("div", {
    class: "llmch-grip",
    title: "Drag this edge to resize the panel",
  });
  grip.addEventListener("pointerdown", (event) => {
    const panel = sidebarPanel();
    if (!panel || event.button !== 0) return;
    event.preventDefault();
    event.stopPropagation();
    const startX = event.clientX;
    const startWidth = panel.getBoundingClientRect().width;
    grip.classList.add("llmch-dragging");
    try { grip.setPointerCapture(event.pointerId); } catch { /* synthetic pointer */ }
    // A live pixel readout makes it obvious that the edge is being dragged.
    const badge = el("span", {
      class: "llmch-grip-badge",
      style: "right:12px;top:50%;transform:translateY(-50%)",
      text: `${Math.round(startWidth)} px`,
    });
    grip.append(badge);

    const onMove = (moveEvent) => {
      const width = applySidebarWidth(startWidth + (moveEvent.clientX - startX), false);
      if (width) badge.textContent = `${width} px`;
      moveEvent.preventDefault();
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      window.removeEventListener("pointercancel", onUp);
      grip.classList.remove("llmch-dragging");
      badge.remove();
      const current = sidebarPanel()?.getBoundingClientRect().width;
      if (current) applySidebarWidth(current);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("pointercancel", onUp);
  });
  return grip;
}

/* ------------------------------------------------------------------ api glue */

async function refreshState() {
  const data = await getJSON(`${BASE}/state`);
  state.settings = data.settings;
  state.conversations = data.conversations || [];
  state.index = data.index || state.index;
  return data;
}

async function openConversation(id) {
  const data = await getJSON(`${BASE}/conversation?id=${encodeURIComponent(id)}`);
  state.conversation = data.conversation;
  state.messages = data.messages || [];
  await postJSON(`${BASE}/settings`, { last_conversation: id });
}

async function newConversation() {
  const settings = state.settings || {};
  const provider = settings.active_provider;
  const info = settings.providers?.[provider] || {};
  let created = null;
  let failure = "";
  try {
    created = await postJSON(`${BASE}/conversation/new`, {
      provider,
      model: info.model || "",
      system_prompt: settings.chat?.system_prompt || settings.default_system_prompt || "",
      rag_enabled: Boolean(settings.rag?.enabled),
      rag_folder: settings.rag?.folder || "",
      rag_top_k: settings.rag?.top_k ?? 4,
      history_limit: settings.chat?.history_limit ?? 20,
      temperature: settings.chat?.temperature ?? 0.7,
      max_tokens: settings.chat?.max_tokens ?? 1024,
    });
  } catch (error) {
    // Still refresh below: the conversation may have been stored even if the
    // answer never came back, and the panel must show what actually exists.
    failure = String(error.message || error);
  }
  await refreshState();
  // Only ever land on the conversation the server returned. Falling back to the
  // current one (or to "the first row") kept the previous chat selected, so the
  // next message was appended to the old conversation instead of a fresh one.
  const target = created?.conversation?.id;
  if (target) {
    await openConversation(target);
    return created.conversation;
  }
  setStatus(
    failure ? `Could not create a conversation: ${failure}` : "Could not create a conversation.",
    true,
  );
  return null;
}

async function patchConversation(fields) {
  if (!state.conversation) return;
  const data = await postJSON(`${BASE}/conversation/update`, { id: state.conversation.id, ...fields });
  state.conversation = data.conversation;
  const found = state.conversations.find((item) => item.id === data.conversation.id);
  if (found) Object.assign(found, data.conversation);
}

/* ------------------------------------------------------------------ rendering */

function messageImages(message) {
  const images = message.images || [];
  if (!images.length) return null;
  return el("div", { class: "llmch-thumbs" }, images.map((image) =>
    el("img", {
      src: api.apiURL(image.url),
      title: image.name,
      onclick: () => window.open(api.apiURL(image.url), "_blank"),
    })));
}

function messageActions(message) {
  const actions = [];
  if (message.content) {
    actions.push(el("button", {
      class: "llmch-btn", text: "Copy",
      onclick: () => navigator.clipboard?.writeText(message.content),
    }));
  }
  if (message.role === "user") {
    actions.push(el("button", {
      class: "llmch-btn", text: "Edit",
      onclick: () => startEdit(message),
    }));
  }
  if (message.role === "assistant") {
    actions.push(el("button", {
      class: "llmch-btn", text: "Regenerate",
      disabled: state.streaming,
      onclick: () => regenerate(message.id),
    }));
  }
  actions.push(el("button", {
    class: "llmch-btn", text: "Delete",
    onclick: () => deleteMessage(message, false),
  }));
  actions.push(el("button", {
    class: "llmch-btn danger", text: "Truncate here",
    title: "Deletes this message and everything after it",
    onclick: () => deleteMessage(message, true),
  }));
  return el("div", { class: "llmch-msg-actions" }, actions);
}

function messageNode(message) {
  const rag = (message.meta?.rag || []).filter(Boolean);
  const head = el("div", { class: "llmch-msg-head" }, [
    el("strong", { text: message.role === "user" ? "You" : message.role === "assistant" ? "Assistant" : message.role }),
    message.edited ? el("span", { text: "• edited" }) : null,
    message.meta?.model ? el("span", { text: `• ${message.meta.model}` }) : null,
    message.meta?.seconds ? el("span", { text: `• ${message.meta.seconds}s` }) : null,
    rag.length ? el("span", { class: "llmch-badge", text: `RAG: ${rag.join(", ")}` }) : null,
  ]);
  const reasoning = message.meta?.reasoning || "";
  const empty = !(message.content || "").trim();
  const body = el("div", {
    class: "llmch-msg-body",
    id: message.id ? `edit-${message.id}` : null,
    html: renderMarkdown(message.content) ||
      (reasoning ? '<span class="llmch-note">The model reasoned without replying: raise "Max tokens".</span>'
                 : empty && message.role === "assistant" ? '<span class="llmch-note">(empty reply)</span>' : ""),
  });
  const images = messageImages(message);
  const thinking = reasoning
    ? el("details", { class: "llmch-reasoning" }, [
        el("summary", { text: "Model reasoning" }),
        el("div", { class: "llmch-think", text: reasoning }),
      ])
    : null;
  const node = el("div", { class: `llmch-msg ${message.role}` }, [head, images, thinking, body]);
  if (message.id) node.append(messageActions(message));
  return node;
}

async function startEdit(message) {
  const holder = document.getElementById(`edit-${message.id}`);
  if (!holder) return;
  const textarea = el("textarea", { class: "llmch-textarea", rows: 4 });
  textarea.value = message.content;
  const box = el("div", { class: "llmch-section" }, [
    textarea,
    el("div", { class: "llmch-row" }, [
      el("button", {
        class: "llmch-btn primary", text: "Save + resend",
        onclick: async () => {
          await postJSON(`${BASE}/messages/edit`, {
            conversation_id: state.conversation.id,
            message_id: message.id,
            content: textarea.value,
          });
          await reloadMessages();
          await regenerate(null);
        },
      }),
      el("button", {
        class: "llmch-btn", text: "Save",
        onclick: async () => {
          await postJSON(`${BASE}/messages/edit`, {
            conversation_id: state.conversation.id,
            message_id: message.id,
            content: textarea.value,
          });
          await reloadMessages();
        },
      }),
      el("button", { class: "llmch-btn", text: "Cancel", onclick: () => renderMessages() }),
    ]),
  ]);
  holder.replaceWith(box);
}

async function deleteMessage(message, fromHere) {
  if (!confirm(fromHere ? "Delete this message and everything after it?" : "Delete this message?")) return;
  const data = await postJSON(`${BASE}/messages/delete`, {
    conversation_id: state.conversation.id,
    message_id: message.id,
    from_here: Boolean(fromHere),
  });
  state.messages = data.messages || [];
  renderMessages();
}

async function reloadMessages() {
  if (!state.conversation) return;
  const data = await getJSON(`${BASE}/conversation?id=${encodeURIComponent(state.conversation.id)}`);
  state.messages = data.messages || [];
  state.conversation = data.conversation;
  renderMessages();
}

/* ------------------------------------------------------------------ chat actions */

function renderStreamingBubble(text, reasoning) {
  // The user may have opened or created another conversation meanwhile: the
  // running reply then belongs to the previous chat and must not be painted here.
  if (state.streamingId && state.conversation && state.streamingId !== state.conversation.id) return;
  const container = rootEl?.querySelector(".llmch-messages");
  if (!container) return;
  let bubble = container.querySelector(".llmch-streaming");
  if (!bubble) {
    bubble = el("div", { class: "llmch-msg assistant llmch-streaming" }, [
      el("div", { class: "llmch-msg-head" }, [el("strong", { text: "Assistant" })]),
      el("details", { class: "llmch-reasoning", style: "display:none" }, [
        el("summary", { text: "Model reasoning" }),
        el("div", { class: "llmch-think" }),
      ]),
      el("div", { class: "llmch-msg-body llmch-caret" }),
    ]);
    container.append(bubble);
  }
  if (reasoning) {
    const holder = bubble.querySelector(".llmch-reasoning");
    holder.style.display = "block";
    holder.querySelector(".llmch-think").textContent = reasoning;
  }
  bubble.querySelector(".llmch-msg-body").innerHTML =
    renderMarkdown(text) || (reasoning ? '<span class="llmch-note">reasoning in progress...</span>' : "");
  container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
  const task = (state.draft || "").trim();
  if (!task && !state.pending.length) return;
  if (state.streaming) return;
  if (!state.conversation) await newConversation();
  if (!state.conversation) return;

  const images = state.pending.map((item) => item.dataUrl);
  state.pending = [];
  state.draft = "";
  renderControls();
  setStatus("Sending...");
  state.streaming = true;
  state.streamingId = state.conversation?.id || null;
  state.controller = new AbortController();
  renderControls();

  let streamed = "";
  let reasoned = "";
  try {
    await streamRequest(`${BASE}/chat`, {
      conversation_id: state.conversation.id,
      text: task,
      images,
    }, {
      onDelta: (delta) => { streamed += delta; renderStreamingBubble(streamed, reasoned); },
      onReasoning: (delta) => { reasoned += delta; renderStreamingBubble(streamed, reasoned); },
      onResult: (result) => {
        if (result.error && !result.reply) setStatus(result.error, true);
        else if (!result.reply && result.reasoning) {
          setStatus("The model reasoned without replying: raise \"Max tokens\" in the settings.", true);
        } else {
          setStatus(result.rag?.length ? `Reply received (RAG: ${result.rag.length} excerpt(s))` : "Reply received");
        }
      },
      onError: (message) => setStatus(message, true),
    }, state.controller.signal);
  } catch (error) {
    if (error.name !== "AbortError") setStatus(String(error.message || error), true);
  } finally {
    state.streaming = false;
    state.streamingId = null;
    state.controller = null;
    await reloadMessages();
    await refreshState();
    renderAll();
  }
}

async function regenerate(messageId) {
  if (state.streaming) return;
  if (!state.conversation) return;
  state.streaming = true;
  state.streamingId = state.conversation.id;
  state.controller = new AbortController();
  renderControls();
  setStatus("Regenerating...");
  let streamed = "";
  let reasoned = "";
  try {
    await streamRequest(`${BASE}/regenerate`, {
      conversation_id: state.conversation.id,
      message_id: messageId || null,
    }, {
      onDelta: (delta) => { streamed += delta; renderStreamingBubble(streamed, reasoned); },
      onReasoning: (delta) => { reasoned += delta; renderStreamingBubble(streamed, reasoned); },
      onError: (message) => setStatus(message, true),
    }, state.controller.signal);
  } catch (error) {
    if (error.name !== "AbortError") setStatus(String(error.message || error), true);
  } finally {
    state.streaming = false;
    state.streamingId = null;
    state.controller = null;
    await reloadMessages();
    renderAll();
  }
}

async function stopStreaming() {
  state.controller?.abort();
  try { await postJSON(`${BASE}/cancel`, { conversation_id: state.conversation?.id }); } catch { /* ignored */ }
  setStatus("Stopped");
}

/* ------------------------------------------------------------------ composer */

function renderComposer() {
  const textarea = el("textarea", {
    class: "llmch-textarea", placeholder: "Write your message... (Enter = send, Shift+Enter = new line)",
    rows: 3,
  });
  textarea.value = state.draft || "";
  // A height chosen with the top grip (or in a previous session) wins over rows=3.
  if (!state.composerHeight) state.composerHeight = savedComposerHeight();
  if (state.composerHeight) applyComposerHeight(textarea, state.composerHeight);
  textarea.addEventListener("input", () => { state.draft = textarea.value; });
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });

  const fileInput = el("input", {
    type: "file", accept: "image/*", multiple: true, style: "display:none",
    onchange: async (event) => {
      for (const file of event.target.files) {
        const dataUrl = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.readAsDataURL(file);
        });
        state.pending.push({ name: file.name, dataUrl });
      }
      event.target.value = "";
      renderComposer();
      const box = rootEl?.querySelector(".llmch-composer textarea");
      box && box.focus();
    },
  });

  const thumbs = state.pending.length
    ? el("div", { class: "llmch-thumbs" }, state.pending.map((item, index) =>
        el("img", {
          src: item.dataUrl, title: `${item.name} (click to remove)`,
          onclick: () => { state.pending.splice(index, 1); renderComposer(); },
        })))
    : null;

  const sendButton = state.streaming
    ? el("button", { class: "llmch-btn danger", text: "Stop", onclick: stopStreaming })
    : el("button", { class: "llmch-btn primary", text: "Send", onclick: sendMessage });

  return el("div", { class: "llmch-composer llmch-section" }, [
    thumbs,
    composerResizeGrip(textarea),
    textarea,
    el("div", { class: "llmch-row" }, [
      el("button", { class: "llmch-btn", text: "Image", onclick: () => fileInput.click() }),
      fileInput,
      el("button", {
        class: "llmch-btn", text: "Clear chat",
        onclick: async () => {
          if (!state.conversation) return;
          if (!confirm("Clear the history of this conversation?")) return;
          await postJSON(`${BASE}/messages/truncate`, { conversation_id: state.conversation.id, keep: 0 });
          await reloadMessages();
        },
      }),
      el("div", { class: "llmch-grow" }),
      sendButton,
    ]),
  ]);
}

/* ------------------------------------------------------------------ header + settings */

function renderHeader() {
  const picker = el("select", { class: "llmch-select llmch-grow" }, state.conversations.map((conversation) =>
    el("option", {
      value: conversation.id,
      text: `${conversation.title || "Untitled"} (${conversation.message_count || 0})`,
      selected: state.conversation?.id === conversation.id,
    })));
  picker.addEventListener("change", async () => {
    await openConversation(picker.value);
    renderAll();
  });

  const provider = PROVIDERS.find((item) => item.value === state.conversation?.provider);
  const model = state.conversation?.model || "(no model selected)";
  const info = state.settings?.providers?.[state.conversation?.provider] || {};

  const ragButton = el("button", {
    class: `llmch-chip ${state.conversation?.rag_enabled ? "on" : ""}`,
    text: state.conversation?.rag_enabled ? "RAG on" : "RAG off",
    title: "Inject document excerpts into every turn of this conversation",
    disabled: !state.conversation,
    onclick: async (event) => {
      const button = event.currentTarget;
      await patchConversation({ rag_enabled: !state.conversation.rag_enabled });
      const active = Boolean(state.conversation.rag_enabled);
      button.textContent = active ? "RAG on" : "RAG off";
      button.className = `llmch-chip ${active ? "on" : ""}`;
      setStatus(active ? "RAG enabled for this conversation." : "RAG disabled.");
    },
  });

  const stats = state.index || {};
  const indexChip = el("span", {
    class: "llmch-chip",
    text: `index: ${stats.documents || 0} docs / ${stats.chunks || 0} excerpts`,
    title: stats.folder ? `Last folder: ${stats.folder}` : "No indexed document",
  });

  return el("div", { class: "llmch-section" }, [
    el("div", { class: "llmch-row" }, [
      picker,
      el("button", {
        class: "llmch-btn", text: "+ New",
        onclick: async () => {
          try {
            await newConversation();
            if (!state.error) setStatus("New conversation.");
          } catch (error) {
            setStatus(String(error.message || error), true);
          }
          renderAll();
        },
      }),
      el("button", {
        class: "llmch-btn", text: "Rename",
        disabled: !state.conversation,
        onclick: async () => {
          const title = prompt("New title", state.conversation?.title || "");
          if (title === null) return;
          await patchConversation({ title });
          await refreshState();
          renderAll();
        },
      }),
      el("button", {
        class: "llmch-btn danger", text: "Delete",
        disabled: !state.conversation,
        title: "Delete the conversation",
        onclick: async () => {
          if (!confirm("Delete this conversation permanently?")) return;
          await postJSON(`${BASE}/conversation/delete`, { id: state.conversation.id });
          state.conversation = null;
          state.messages = [];
          await refreshState();
          if (state.conversations.length) await openConversation(state.conversations[0].id);
          renderAll();
        },
      }),
      el("button", {
        class: "llmch-btn", text: state.showPanel ? "Hide settings" : "Settings",
        onclick: () => { state.showPanel = !state.showPanel; renderAll(); },
      }),
    ]),
    el("div", { class: "llmch-row llmch-chips" }, [
      ragButton,
      indexChip,
      el("span", {
        class: "llmch-chip",
        id: "llmch-provider-chip",
        text: `${provider?.label || state.conversation?.provider || "-"} / ${model}`,
        title: `URL: ${info.base_url || "-"}`,
      }),
    ]),
  ]);
}

function settingsPanel() {
  const settings = state.settings || {};
  const conversation = state.conversation;
  const activeProvider = conversation?.provider || settings.active_provider;

  const providerSelect = el("select", { class: "llmch-select" }, PROVIDERS.map((item) =>
    el("option", { value: item.value, text: item.label, selected: item.value === activeProvider })));

  const providerInfo = settings.providers?.[activeProvider] || {};
  const baseUrl = el("input", { class: "llmch-input", placeholder: "Base URL", value: providerInfo.base_url || "" });
  // A datalist keeps the field free-form (a model missing from the server list can
  // still be typed) while offering every model the provider reports.
  const modelOptions = el("datalist", { id: "llmch-model-options" });
  const model = el("input", {
    class: "llmch-input",
    placeholder: "model (pick one in the list or type a name)",
    list: "llmch-model-options",
    value: conversation?.model || providerInfo.model || "",
  });
  const apiKey = el("input", { class: "llmch-input", type: "password", placeholder: providerInfo.has_key ? "(key saved)" : "API key (Ollama Cloud)" });

  const temperature = el("input", { class: "llmch-input", type: "number", step: "0.05", min: "0", max: "2", value: conversation?.temperature ?? settings.chat?.temperature ?? 0.7 });
  const maxTokens = el("input", { class: "llmch-input", type: "number", min: "16", max: "131072", value: conversation?.max_tokens ?? settings.chat?.max_tokens ?? 1024 });
  const historyLimit = el("input", { class: "llmch-input", type: "number", min: "0", max: "200", value: conversation?.history_limit ?? settings.chat?.history_limit ?? 20 });

  const systemPrompt = el("textarea", { class: "llmch-textarea", rows: 3, placeholder: "Starting prompt / system instructions" });
  systemPrompt.value = conversation?.system_prompt || settings.chat?.system_prompt || "";
  const reference = el("textarea", { class: "llmch-textarea", rows: 2, placeholder: "Reference context, always included (free text)" });
  reference.value = conversation?.reference || "";

  const ragFolder = el("input", { class: "llmch-input", placeholder: "Folder of documents to index", value: conversation?.rag_folder || settings.rag?.folder || "" });
  const ragTopK = el("input", { class: "llmch-input", type: "number", min: "1", max: "20", value: conversation?.rag_top_k ?? settings.rag?.top_k ?? 4 });
  const embedModel = el("input", { class: "llmch-input", placeholder: "embedding model (empty = chat model)", value: settings.rag?.embed_model || "" });

  function applyModels(models, target) {
    modelOptions.replaceChildren(...models.map((name) => el("option", { value: name })));
    // A brand-new conversation has no model yet: pick the first one so the panel is
    // usable straight away, and say so rather than choosing silently.
    if (models.length && !model.value.trim()) {
      model.value = models[0];
      return `kept "${models[0]}"`;
    }
    return "";
  }

  /** Fill the model list of a provider, from the cache unless `force` is set. */
  async function fetchModels({ announce = true, force = false, provider, base, key } = {}) {
    const target = provider || providerSelect.value;
    const baseValue = base === undefined ? baseUrl.value : base;
    const cacheKey = `${target}|${baseValue}`;
    const cached = state.modelCache[cacheKey];
    if (!force && cached) {
      applyModels(cached, target);
      if (announce) setStatus(`${cached.length} model(s) on ${target}.`);
      return cached;
    }
    if (announce) setStatus(`Fetching models from ${target}...`);
    try {
      const data = await postJSON(`${BASE}/models`, {
        provider: target,
        base_url: baseValue,
        api_key: key === undefined ? apiKey.value : key,
      });
      const models = (data.models || []).filter(Boolean);
      state.modelCache[cacheKey] = models;
      const picked = applyModels(models, target);
      if (!models.length) setStatus(`No model listed by ${target}.`, announce);
      else if (announce || picked) setStatus(`${models.length} model(s) on ${target}${picked ? ` - ${picked}` : ""}.`);
      return models;
    } catch (error) {
      delete state.modelCache[cacheKey];
      modelOptions.replaceChildren();
      setStatus(String(error.message || error), announce);
      return [];
    }
  }

  providerSelect.addEventListener("change", async () => {
    const chosen = providerSelect.value;
    const chosenLabel = PROVIDERS.find((item) => item.value === chosen)?.label || chosen;
    const info = settings.providers?.[chosen] || {};
    baseUrl.value = info.base_url || "";
    model.value = info.model || "";
    modelOptions.replaceChildren();
    const models = await fetchModels({ announce: false, force: true, provider: chosen });
    if (!model.value.trim() && models.length) model.value = models[0];
    // The picker is the source of truth, immediately: remember the choice for the
    // next conversations *and* move the open one, otherwise leaving the tab and
    // coming back would silently restore the previous backend.
    try {
      const data = await postJSON(`${BASE}/settings`, {
        active_provider: chosen,
        // Remember the model shown here, so "+ New" starts on it too.
        providers: { [chosen]: { model: model.value.trim() } },
      });
      state.settings = data.settings;
      if (state.conversation) {
        await patchConversation({ provider: chosen, model: model.value.trim() });
      }
      const chip = rootEl?.querySelector("#llmch-provider-chip");
      if (chip) {
        chip.textContent = `${chosenLabel} / ${model.value.trim() || "(no model selected)"}`;
        chip.title = `URL: ${baseUrl.value || "-"}`;
      }
      setStatus(
        `${models.length} model(s) on ${chosen}` +
        (state.conversation ? ` - this conversation now uses ${model.value.trim() || "(no model yet)"}.` : "."),
      );
    } catch (error) {
      setStatus(String(error.message || error), true);
    }
  });

  const loadModels = el("button", {
    class: "llmch-btn", text: "Reload models",
    title: "Ask the selected provider for its model list",
    onclick: () => fetchModels({ announce: true, force: true }),
  });

  const testConnection = el("button", {
    class: "llmch-btn", text: "Test connection",
    onclick: async () => {
      setStatus("Testing...");
      try {
        const data = await postJSON(`${BASE}/models`, {
          provider: providerSelect.value, base_url: baseUrl.value, api_key: apiKey.value,
        });
        setStatus(`OK - ${data.models.length} model(s) on ${providerSelect.value}.`);
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    },
  });

  const saveConnection = el("button", {
    class: "llmch-btn primary", text: "Save",
    onclick: async () => {
      setStatus("Saving...");
      try {
        const patch = {
          active_provider: providerSelect.value,
          providers: {
            [providerSelect.value]: {
              base_url: baseUrl.value.trim(),
              model: model.value.trim(),
              ...(apiKey.value.trim() ? { api_key: apiKey.value.trim() } : {}),
            },
          },
          chat: {
            temperature: Number(temperature.value) || 0.7,
            max_tokens: Number(maxTokens.value) || 1024,
            history_limit: Number(historyLimit.value) || 0,
            system_prompt: systemPrompt.value,
          },
          rag: {
            folder: ragFolder.value.trim(),
            top_k: Number(ragTopK.value) || 4,
            embed_model: embedModel.value.trim(),
            enabled: Boolean(state.conversation?.rag_enabled),
          },
        };
        const data = await postJSON(`${BASE}/settings`, patch);
        state.settings = data.settings;
        if (state.conversation) {
          await patchConversation({
            provider: providerSelect.value,
            model: model.value.trim(),
            temperature: Number(temperature.value) || 0.7,
            max_tokens: Number(maxTokens.value) || 1024,
            history_limit: Number(historyLimit.value) || 0,
            system_prompt: systemPrompt.value,
            reference: reference.value,
            rag_folder: ragFolder.value.trim(),
            rag_top_k: Number(ragTopK.value) || 4,
          });
        }
        apiKey.value = "";
        setStatus(`Settings saved for ${providerSelect.value}.`);
        renderAll();
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    },
  });

  async function runIndex() {
    const folder = ragFolder.value.trim();
    if (!folder) { setStatus("Add a document or set a folder to index.", true); return; }
    setStatus("Indexing...");
    try {
      await streamRequest(`${BASE}/rag/index`, {
        folder,
        provider: providerSelect.value,
        base_url: baseUrl.value,
        embed_model: embedModel.value.trim(),
      }, {
        onProgress: (progress) => setStatus(`${progress.stage}: ${progress.done}/${progress.total} - ${progress.message}`),
        onResult: (summary) => {
          setStatus(
            `Index complete: ${summary.documents} document(s), ${summary.chunks} excerpt(s)` +
            (summary.embedding_warning ? `\nEmbeddings unavailable, keyword fallback: ${summary.embedding_warning}` : ""),
            Boolean(summary.embedding_warning),
          );
          state.index = summary.stats || state.index;
        },
        onError: (message) => setStatus(message, true),
      });
      await refreshState();
      renderAll();
    } catch (error) {
      setStatus(String(error.message || error), true);
    }
  }

  const indexButton = el("button", { class: "llmch-btn", text: "Index documents", onclick: () => runIndex() });

  const filePicker = el("input", {
    type: "file",
    multiple: true,
    style: "display:none",
    accept: TEXT_ACCEPT,
  });
  const fileList = el("div", { class: "llmch-files" });

  function renderFileList() {
    const files = state.documents || [];
    if (!files.length) {
      fileList.replaceChildren(el("div", { class: "llmch-label", text: "No document imported." }));
      return;
    }
    fileList.replaceChildren(...files.map((file) => el("div", { class: "llmch-file" }, [
      el("span", { class: "llmch-file-name", text: file.name, title: file.name }),
      el("span", { class: "llmch-file-size", text: formatSize(file.size) }),
      el("button", { class: "llmch-btn danger", text: "x", title: `Remove ${file.name}`, onclick: () => removeFile(file.name) }),
    ])));
  }

  async function refreshFiles() {
    try {
      const data = await getJSON(`${BASE}/rag/files`);
      state.documents = data.files || [];
    } catch (error) {
      state.documents = state.documents || [];
    }
    renderFileList();
  }

  async function removeFile(name) {
    try {
      const data = await postJSON(`${BASE}/rag/files/delete`, { name });
      state.documents = data.files || [];
      renderFileList();
      setStatus(`${name} removed from the folder. Reindex to refresh the index.`);
    } catch (error) {
      setStatus(String(error.message || error), true);
    }
  }

  async function uploadDocuments(files) {
    if (!files.length) return;
    setStatus(`Importing ${files.length} file(s)...`);
    try {
      const payload = [];
      for (const file of files) payload.push({ name: file.name, data: await readFileBase64(file) });
      const data = await postJSON(`${BASE}/rag/files`, { files: payload });
      state.documents = data.files || [];
      renderFileList();
      if (data.folder) ragFolder.value = data.folder;
      await postJSON(`${BASE}/settings`, { rag: { folder: ragFolder.value.trim() } });
      if (state.conversation) await patchConversation({ rag_folder: ragFolder.value.trim() });
      if ((data.skipped || []).length) {
        setStatus(`Imported: ${(data.saved || []).join(", ")}\nSkipped: ${data.skipped.join(", ")}`, true);
      } else {
        setStatus(`${(data.saved || []).length} document(s) imported. Indexing...`);
      }
      await runIndex();
    } catch (error) {
      setStatus(String(error.message || error), true);
    }
  }

  filePicker.addEventListener("change", () => {
    const picked = Array.from(filePicker.files || []);
    filePicker.value = "";
    uploadDocuments(picked);
  });

  const clearIndex = el("button", {
    class: "llmch-btn danger", text: "Clear index",
    onclick: async () => {
      if (!confirm("Delete the whole document index?")) return;
      const data = await postJSON(`${BASE}/rag/clear`, {});
      state.index = data.index || state.index;
      renderAll();
    },
  });

  const testRag = el("button", {
    class: "llmch-btn", text: "Test RAG",
    onclick: async () => {
      const query = prompt("RAG test query", "document summary");
      if (!query) return;
      try {
        const data = await postJSON(`${BASE}/rag/test`, { query, provider: providerSelect.value });
        const lines = (data.results || []).map((result) =>
          `${(result.source || "").split(/[\\/]/).pop()} (${result.score}, ${result.mode}): ${result.text.slice(0, 140)}...`);
        setStatus(lines.length ? lines.join("\n\n") : "No relevant excerpt.");
      } catch (error) {
        setStatus(String(error.message || error), true);
      }
    },
  });

  const panel = el("div", { class: "llmch-panel" }, [
    el("div", { class: "llmch-section" }, [
      el("div", { class: "llmch-label", text: "Connection (global)" }),
      providerSelect,
      baseUrl,
      el("div", { class: "llmch-row" }, [model, modelOptions, loadModels]),
      apiKey,
      el("div", { class: "llmch-row" }, [
        el("label", { class: "llmch-grow" }, [el("div", { class: "llmch-label", text: "Temperature" }), temperature]),
        el("label", { class: "llmch-grow" }, [el("div", { class: "llmch-label", text: "Max tokens" }), maxTokens]),
        el("label", { class: "llmch-grow" }, [el("div", { class: "llmch-label", text: "History turns" }), historyLimit]),
      ]),
      el("div", { class: "llmch-row" }, [testConnection, saveConnection]),
    ]),
    el("div", { class: "llmch-section" }, [
      el("div", { class: "llmch-label", text: "Starting prompt (conversation)" }),
      systemPrompt,
      el("div", { class: "llmch-label", text: "Reference context" }),
      reference,
    ]),
    el("div", { class: "llmch-section" }, [
      el("div", {
        class: "llmch-label",
        text: `RAG documents - index: ${state.index?.documents || 0} doc / ${state.index?.chunks || 0} excerpts`,
      }),
      el("div", { class: "llmch-row" }, [
        el("button", { class: "llmch-btn primary", text: "Add documents", onclick: () => filePicker.click() }),
        filePicker,
      ]),
      fileList,
      ragFolder,
      el("div", { class: "llmch-row" }, [
        el("label", { class: "llmch-grow" }, [el("div", { class: "llmch-label", text: "Excerpts (top-k)" }), ragTopK]),
        el("label", { class: "llmch-grow" }, [el("div", { class: "llmch-label", text: "Embedding model" }), embedModel]),
      ]),
      el("div", { class: "llmch-row" }, [indexButton, testRag, clearIndex]),
    ]),
  ]);

  refreshFiles();
  fetchModels({ announce: false });

  return panel;
}

/* ------------------------------------------------------------------ assembly */

function setStatus(text, isError = false) {
  state.status = text || "";
  state.error = isError ? text : "";
  const bar = rootEl?.querySelector(".llmch-status");
  if (bar) {
    bar.textContent = state.status;
    bar.className = `llmch-status${isError ? " error" : ""}`;
  }
}

function renderMessages() {
  const container = rootEl?.querySelector(".llmch-messages");
  if (!container) return;
  if (!state.messages.length) {
    container.replaceChildren(el("div", {
      class: "llmch-empty",
      text: "No message yet. Write your starting prompt, attach an image if needed, and send.",
    }));
    return;
  }
  container.replaceChildren(...state.messages.map(messageNode));
  container.scrollTop = container.scrollHeight;
}

function renderControls() {
  const composer = rootEl?.querySelector(".llmch-composer");
  if (composer) composer.replaceWith(renderComposer());
}

function renderAll() {
  if (!rootEl) return;
  rootEl.classList.toggle("settings-open", Boolean(state.showPanel));
  const scroll = rootEl.querySelector(".llmch-messages")?.scrollTop ?? 0;
  const nodes = [
    renderHeader(),
    state.showPanel ? settingsPanel() : null,
    el("div", { class: "llmch-messages" }),
    renderComposer(),
    el("div", { class: `llmch-status${state.error ? " error" : ""}`, text: state.status }),
    resizeGrip(),
  ];
  rootEl.replaceChildren(...nodes.filter(Boolean));
  renderMessages();
  const container = rootEl.querySelector(".llmch-messages");
  if (container) container.scrollTop = scroll;
}

/**
 * Refresh the panel after a graph run.
 *
 * A node can open or extend a conversation; without this the panel stayed frozen on
 * whatever it showed last, and the new conversation was only visible after leaving
 * the tab and coming back.
 */
async function syncAfterGraphRun() {
  if (!rootEl || state.streaming) return;
  try {
    const before = state.conversations.map((c) => `${c.id}:${c.message_count}:${c.updated_at}`).join("|");
    const known = new Set(state.conversations.map((c) => c.id));
    await refreshState();
    const after = state.conversations.map((c) => `${c.id}:${c.message_count}:${c.updated_at}`).join("|");
    if (!after || before === after) return; // Nothing new: keep the view, and the focus.
    // Only a conversation that did not exist before the run may take the view. A
    // run that merely extended an older chat must not steal the panel away from
    // the empty chat the user just opened with "+ New" (that is how a new chat
    // used to end up writing into the previous conversation).
    const appeared = state.conversations.find((c) => !known.has(c.id));
    const current = state.conversation?.id;
    if (appeared) await openConversation(appeared.id);
    else if (current && state.conversations.some((c) => c.id === current)) await reloadMessages();
    else if (state.conversations[0]) await openConversation(state.conversations[0].id);
    renderAll();
    const shown = state.conversations.find((c) => c.id === state.conversation?.id);
    if (shown) setStatus(`Graph run: "${shown.title}" (${shown.message_count} message(s)).`);
  } catch (error) {
    // The panel is only a viewer here: a failed sync must not disturb the chat.
    console.warn("[LLM ChatHub] refresh after graph run failed", error);
  }
}

async function mount(target) {
  injectCss();
  rootEl = el("div", { class: "llmch-root" });
  rootEl.replaceChildren(el("div", { class: "llmch-empty", text: "Loading ChatHub..." }));
  target.append(rootEl);
  try {
    const data = await refreshState();
    const wanted = state.settings?.last_conversation;
    const first = data.conversations.find((item) => item.id === wanted) || data.conversations[0];
    if (first) await openConversation(first.id);
    else await newConversation();
    renderAll();
    // The width chosen with the grip survives a tab switch and a ComfyUI restart.
    restoreSidebarWidth();
    setStatus("Ready.");
  } catch (error) {
    rootEl.replaceChildren(el("div", { class: "llmch-empty", text: `ChatHub unavailable: ${error.message}` }));
  }
}

/* ------------------------------------------------------------------ node: RAG upload */

function notify(summary, detail = "", severity = "success") {
  try {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
      toast.add({ severity, summary, detail, life: severity === "error" ? 8000 : 5000 });
      return;
    }
  } catch { /* no toast API in this version */ }
  console.log(`[LLM ChatHub] ${summary}${detail ? ` - ${detail}` : ""}`);
}

/** Write a value into a widget while keeping ComfyUI's screen value in sync. */
function setWidgetValue(node, widget, value) {
  if (!widget) return false;
  widget.value = value;
  const element = widget.element || null;
  if (element) {
    if (element.type === "checkbox") element.checked = Boolean(value);
    else if ("value" in element) element.value = value;
  }
  try { widget.callback?.(value); } catch { /* user callback */ }
  try { node.setDirtyCanvas?.(true, true); } catch { /* noop */ }
  return true;
}

function attachRagUpload(node) {
  if (!node?.widgets || node.__llmchRagUpload) return;
  const folderWidget = node.widgets.find((widget) => widget.name === "rag_folder");
  if (!folderWidget) return;
  node.__llmchRagUpload = true;

  const label = "RAG upload (files)";
  const picker = document.createElement("input");
  picker.type = "file";
  picker.multiple = true;
  picker.accept = TEXT_ACCEPT;
  picker.style.display = "none";
  document.body.append(picker);

  const button = node.addWidget("button", label, null, () => {
    picker.value = "";
    picker.click();
  }, { serialize: false });
  if (!button) {
    picker.remove();
    node.__llmchRagUpload = false;
    return;
  }

  function setLabel(text) {
    button.name = text;
    try { node.setDirtyCanvas?.(true, true); } catch { /* noop */ }
  }

  picker.addEventListener("change", async () => {
    const files = Array.from(picker.files || []);
    if (!files.length) return;
    setLabel("RAG: import...");
    try {
      const payload = [];
      for (const file of files) payload.push({ name: file.name, data: await readFileBase64(file) });
      const uploaded = await postJSON(`${BASE}/rag/files`, { files: payload });
      const saved = uploaded.saved || [];
      const skipped = uploaded.skipped || [];
      if (!saved.length) {
        notify("RAG: no file imported", skipped.join(", ") || "unsupported format", "error");
        return;
      }

      setWidgetValue(node, folderWidget, uploaded.folder || "");
      setWidgetValue(node, node.widgets.find((widget) => widget.name === "rag_enabled"), true);

      setLabel("RAG: indexing...");
      let failure = "";
      let summary = null;
      await streamRequest(`${BASE}/rag/index`, { folder: uploaded.folder || folderWidget.value }, {
        onProgress: (progress) => setLabel(`RAG: ${progress.done}/${progress.total}`),
        onResult: (result) => { summary = result; },
        onError: (message) => { failure = typeof message === "string" ? message : JSON.stringify(message); },
      });

      if (failure || !summary) {
        notify("RAG: indexing failed", failure || "no result", "error");
        return;
      }
      const detail = `${summary.documents} document(s), ${summary.chunks} excerpt(s)` +
        (summary.embedding_warning ? ` - embeddings unavailable: ${summary.embedding_warning}` : "");
      notify("RAG indexed", detail, summary.embedding_warning ? "warn" : "success");
      if (skipped.length) notify("RAG: skipped files", skipped.join(", "), "warn");
    } catch (error) {
      notify("RAG: error", String(error?.message || error), "error");
    } finally {
      setLabel(label);
    }
  });
}

/**
 * Let a node point at a stored conversation.
 *
 * The node creates its own chat (``auto``), but a graph can also continue a
 * conversation written from the panel: this dialog writes the chosen id into the
 * ``session_id`` widget, so the choice is saved in the workflow itself.
 */
async function openConversationsDialog(node, idWidget) {
  const overlay = el("div", { class: "llmch-modal" });
  const close = () => overlay.remove();
  overlay.addEventListener("mousedown", (event) => { if (event.target === overlay) close(); });

  let list = [];
  try {
    const data = await getJSON(`${BASE}/conversations`);
    list = data.conversations || [];
  } catch { /* keep the empty list: the dialog still offers "+ New chat" */ }

  const current = String(idWidget.value || "auto").trim();
  const select = el("select", { class: "llmch-select" }, list.map((conversation) =>
    el("option", {
      value: conversation.id,
      text: `${conversation.title || "Untitled"} (${conversation.message_count || 0}) - ${conversation.model || conversation.provider || "?"}`,
      selected: conversation.id === current,
    })));
  const status = el("div", { class: "llmch-status" });
  const settings = state.settings || {};
  const activeProvider = settings.active_provider || "";

  const useButton = el("button", {
    class: "llmch-btn primary",
    text: "Use this conversation",
    disabled: !list.length,
    onclick: () => {
      if (!select.value) return;
      const label = select.options[select.selectedIndex]?.text || select.value;
      setWidgetValue(node, idWidget, select.value);
      notify("Conversation selected", label);
      close();
    },
  });

  const newButton = el("button", {
    class: "llmch-btn",
    text: "+ New chat",
    title: "Create a conversation and point this node at it",
    onclick: async () => {
      newButton.disabled = true;
      status.className = "llmch-status";
      status.textContent = "Creating...";
      try {
        const created = await postJSON(`${BASE}/conversation/new`, {
          provider: activeProvider || undefined,
          model: settings.providers?.[activeProvider]?.model || undefined,
        });
        const id = created?.conversation?.id;
        if (!id) throw new Error("the server did not return a conversation");
        setWidgetValue(node, idWidget, id);
        notify("New conversation", id);
        close();
      } catch (error) {
        status.className = "llmch-status error";
        status.textContent = String(error?.message || error);
        newButton.disabled = false;
      }
    },
  });

  const autoButton = el("button", {
    class: "llmch-btn",
    text: "Back to auto",
    title: "Stop pointing at a stored chat: this node keeps its own conversation",
    onclick: () => {
      setWidgetValue(node, idWidget, "auto");
      notify("Conversation", "This node keeps its own chat again.");
      close();
    },
  });

  overlay.append(el("div", { class: "llmch-modal-body" }, [
    el("div", { class: "llmch-label", text: "Stored conversations" }),
    list.length
      ? select
      : el("div", { class: "llmch-empty", text: "No stored conversation yet." }),
    el("div", { class: "llmch-row" }, [
      useButton,
      newButton,
      autoButton,
      el("div", { class: "llmch-grow" }),
      el("button", { class: "llmch-btn", text: "Cancel", onclick: close }),
    ]),
    status,
    el("div", { class: "llmch-note", text: `Current: ${current || "auto"}` }),
  ]));
  document.body.append(overlay);
}

function attachConversationPicker(node) {
  if (!node?.widgets || node.__llmchConvPicker) return;
  const idWidget = node.widgets.find((widget) => widget.name === "session_id");
  if (!idWidget) return;
  node.__llmchConvPicker = true;
  const button = node.addWidget("button", "Conversations...", null, () => {
    openConversationsDialog(node, idWidget).catch((error) => {
      notify("Conversations", String(error?.message || error), "error");
    });
  }, { serialize: false });
  if (!button) node.__llmchConvPicker = false;
}

app.registerExtension({
  name: "LLM.ChatHub.RagUpload",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!["LLMChatHubSession", "LCSession"].includes(nodeData?.name)) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = onNodeCreated?.apply(this, arguments);
      try {
        attachRagUpload(this);
      } catch (error) {
        console.warn("[LLM ChatHub] RAG upload button unavailable", error);
      }
      try {
        attachConversationPicker(this);
      } catch (error) {
        console.warn("[LLM ChatHub] conversation picker unavailable", error);
      }
      return result;
    };
  },
});

app.registerExtension({
  name: "LLM.ChatHub",
  async setup() {
    injectCss();
    const tab = {
      id: "llm-chathub",
      title: "ChatHub",
      icon: "pi pi-comments",
      type: "custom",
      tooltip: "Chat with LM Studio / Ollama (history, images, RAG)",
      render: (element) => {
        element.style.height = "100%";
        element.style.width = "100%";
        mount(element);
      },
    };
    for (const event of ["execution_success", "execution_error", "execution_interrupted"]) {
      api.addEventListener(event, () => { syncAfterGraphRun(); });
    }
    // Those events are not emitted by every ComfyUI version, so a light poll covers
    // the rest of them. It only costs one request, and only while the tab is on
    // screen: a hidden or closed panel does nothing.
    setInterval(() => {
      // clientWidth is 0 while the tab is hidden: a closed panel costs nothing.
      if (!rootEl?.isConnected || !rootEl.clientWidth) return;
      syncAfterGraphRun();
    }, 4000);
    try {
      app.extensionManager.registerSidebarTab(tab);
    } catch (error) {
      console.warn("[LLM ChatHub] sidebar unavailable, falling back to a floating window", error);
      const overlay = el("div", { class: "llmch-overlay" });
      document.body.append(overlay);
      mount(overlay);
    }
  },
});
