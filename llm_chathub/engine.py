"""Conversation engine: prompt assembly, provider call, persistence.

Both the ComfyUI nodes and the browser panel call :func:`run_chat`, so a
conversation started from the graph can be continued in the panel and the other
way round. The engine is synchronous; the panel drives it from a worker thread.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
import uuid

import numpy as np

from . import config, providers, rag, store

MAX_IMAGES_IN_CONTEXT = 4
TITLE_MAX = 60

_cancels: dict[str, threading.Event] = {}
_cancels_lock = threading.Lock()


# --------------------------------------------------------------------------- images


def tensor_to_base64(image) -> str:
    """Convert one ComfyUI IMAGE tensor (H,W,C float 0..1) to a PNG data payload."""
    from PIL import Image

    array = image
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 4:
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
        array = array.transpose(1, 2, 0)
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def split_image_batch(image) -> list:
    """Split a ComfyUI IMAGE batch into single frames."""
    if image is None:
        return []
    array = image
    if hasattr(array, "shape") and len(getattr(array, "shape", ())) == 4:
        return [array[index] for index in range(array.shape[0])]
    return [array]


def save_base64_image(data: str, suffix: str = ".png") -> str:
    """Persist a base64 image in the user image folder, return its stored name."""
    config.ensure_dirs()
    payload = data.split(",", 1)[1] if data.startswith("data:") else data
    name = f"{uuid.uuid4().hex[:16]}{suffix}"
    path = os.path.join(config.images_dir(), name)
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(payload))
    return name


def image_to_base64(name: str) -> str | None:
    path = os.path.join(config.images_dir(), os.path.basename(name))
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")


# --------------------------------------------------------------------------- cancellation


def cancel_token(conversation_id: str) -> threading.Event:
    with _cancels_lock:
        event = _cancels.get(conversation_id)
        if event is None:
            event = threading.Event()
            _cancels[conversation_id] = event
        return event


def request_cancel(conversation_id: str) -> bool:
    with _cancels_lock:
        event = _cancels.get(conversation_id)
    if event is None:
        return False
    event.set()
    return True


def clear_cancel(conversation_id: str) -> None:
    with _cancels_lock:
        _cancels.pop(conversation_id, None)


# --------------------------------------------------------------------------- prompt assembly


def _reference_block(reference: str) -> str:
    reference = (reference or "").strip()
    if not reference:
        return ""
    return (
        "\n\n--- REFERENCE CONTEXT PROVIDED BY THE USER ---\n"
        f"{reference}\n--- END OF REFERENCE CONTEXT ---"
    )


def _rag_block(results: list[dict]) -> str:
    context = rag.build_context(results)
    if not context:
        return ""
    return (
        "\n\n--- DOCUMENT EXCERPTS (RAG) ---\n"
        "Use these excerpts when they are relevant and cite the file between square brackets.\n\n"
        f"{context}\n--- END OF EXCERPTS ---"
    )


def resolve_conversation(
    conversation_id: str | None = None,
    *,
    create: bool = True,
    **overrides,
) -> dict:
    """Fetch a conversation, creating it (with the current settings) if needed."""
    conversation = store.get_conversation(conversation_id) if conversation_id else None
    if conversation:
        if overrides:
            conversation = store.update_conversation(conversation["id"], **overrides) or conversation
        return conversation
    if not create:
        raise ValueError(f"Unknown conversation: {conversation_id}")
    return store.create_conversation(id=conversation_id or None, **overrides)


def build_messages(
    conversation: dict,
    history: list[dict],
    *,
    query: str,
    query_images: list[str] | None = None,
    rag_results: list[dict] | None = None,
) -> list[dict]:
    system = (conversation.get("system_prompt") or config.DEFAULT_SYSTEM_PROMPT).strip()
    system += _reference_block(conversation.get("reference", ""))
    if rag_results:
        system += _rag_block(rag_results)

    messages = [{"role": "system", "content": system}]
    budget = MAX_IMAGES_IN_CONTEXT
    selected: list[dict] = []
    for message in reversed(history):
        images = list(message.get("images") or [])
        if images:
            if budget <= 0:
                images = []
            elif len(images) > budget:
                images = images[-budget:]
            budget -= len(images)
        selected.append({**message, "images": images})
    for message in reversed(selected):
        if message["role"] == "system":
            continue
        messages.append(
            {
                "role": message["role"],
                "content": message.get("content") or "",
                "images": message["images"],
            }
        )
    # The current turn is never part of the stored history at this point.
    messages.append({"role": "user", "content": query, "images": list(query_images or [])})
    return messages


def _history_for(conversation: dict) -> list[dict]:
    """History as the model should see it.

    Empty answers (a reasoning model that ran out of tokens) and error placeholders
    are kept in the UI but never sent back to the model.
    """
    limit = max(0, int(conversation.get("history_limit", 20)))
    messages = []
    for message in store.list_messages(conversation["id"]):
        if message["role"] == "system":
            continue
        if message["role"] == "assistant":
            if not (message.get("content") or "").strip():
                continue
            if (message.get("meta") or {}).get("error"):
                continue
        messages.append(message)
    if limit:
        messages = messages[-limit:]
    return messages


def _hydrate_images(message: dict) -> dict:
    encoded = []
    for name in message.get("images") or []:
        data = image_to_base64(name)
        if data:
            encoded.append(data)
    return {**message, "images": encoded}


def _maybe_title(conversation: dict, first_text: str, is_first_message: bool) -> dict:
    if is_first_message and first_text.strip():
        title = first_text.strip().splitlines()[0].strip()[:TITLE_MAX]
        if title:
            return store.update_conversation(conversation["id"], title=title) or conversation
    return conversation


# --------------------------------------------------------------------------- main entry


def run_chat(
    conversation_id: str | None,
    text: str,
    *,
    images: list[str] | None = None,
    image_payloads: list[str] | None = None,
    attach_b64: list[str] | None = None,
    on_delta=None,
    on_reasoning=None,
    persist: bool = True,
    connection: dict | None = None,
    create_if_missing: bool = True,
    **overrides,
) -> dict:
    """Send one user turn and return the assistant reply.

    ``images`` are stored file names, ``image_payloads`` are raw base64/data-url
    strings that get saved first (used by the panel uploads), and ``attach_b64``
    are images attached to this turn only (used when history is disabled).
    ``connection`` overrides base_url / api_key / embed_model from stored settings.
    """
    started = time.time()
    conversation = resolve_conversation(
        conversation_id, create=create_if_missing, **overrides
    )
    stored_images = list(images or [])
    for payload in image_payloads or []:
        stored_images.append(save_base64_image(payload))

    existing = store.list_messages(conversation["id"])
    is_first = not existing

    resolved = config.resolve_provider(
        conversation.get("provider"),
        model=conversation.get("model") or None,
    )
    for key in ("base_url", "api_key", "embed_model", "timeout"):
        value = (connection or {}).get(key)
        if value:
            resolved[key] = value

    rag_results: list[dict] = []
    if conversation.get("rag_enabled"):
        try:
            rag_results = rag.retrieve(
                text,
                top_k=int(conversation.get("rag_top_k") or resolved["rag_top_k"]),
                provider=resolved["provider"],
                base_url=resolved["base_url"],
                api_key=resolved["api_key"],
                embed_model=resolved["embed_model"],
            )
        except Exception:
            rag_results = []

    turn_images = [image_to_base64(name) for name in stored_images]
    turn_images = [data for data in turn_images if data]
    turn_images.extend(attach_b64 or [])

    history = [_hydrate_images(message) for message in _history_for(conversation)]
    messages = build_messages(
        conversation, history, query=text, query_images=turn_images, rag_results=rag_results
    )

    if persist:
        conversation = _maybe_title(conversation, text, is_first)
        user_message = store.add_message(
            conversation["id"], "user", text, stored_images or None
        )
    else:
        user_message = {"id": None, "role": "user", "content": text, "images": stored_images}

    cancel = cancel_token(conversation["id"])
    cancel.clear()
    buffer: list[str] = []
    thinking: list[str] = []
    error: str | None = None

    def push_reasoning(text: str) -> None:
        thinking.append(text)
        if on_reasoning:
            on_reasoning(text)

    try:
        for delta in providers.stream_chat(
            resolved["provider"],
            resolved["base_url"],
            conversation.get("model") or resolved["model"],
            messages,
            temperature=float(conversation.get("temperature") or resolved["temperature"]),
            max_tokens=int(conversation.get("max_tokens") or resolved["max_tokens"]),
            api_key=resolved["api_key"],
            timeout=resolved["timeout"],
            cancel=cancel,
            on_reasoning=push_reasoning,
        ):
            buffer.append(delta)
            if on_delta:
                on_delta(delta)
    except providers.ProviderError as exception:
        error = str(exception)
    finally:
        clear_cancel(conversation["id"])

    reply = "".join(buffer).strip()
    reasoning = "".join(thinking).strip()
    cancelled = cancel.is_set()

    assistant_message = None
    if persist and (reply or error or reasoning):
        assistant_message = store.add_message(
            conversation["id"],
            "assistant",
            reply if (reply or not error) else f"[error] {error}",
            None,
            meta={
                "provider": resolved["provider"],
                "model": conversation.get("model") or resolved["model"],
                "seconds": round(time.time() - started, 2),
                "rag": [os.path.basename(r.get("source") or "") for r in rag_results],
                "cancelled": cancelled,
                "error": error,
                "reasoning": reasoning[:20000],
                "reasoning_only": bool(reasoning and not reply),
            },
        )

    return {
        "conversation_id": conversation["id"],
        "title": conversation.get("title"),
        "reply": reply,
        "reasoning": reasoning,
        "error": error,
        "cancelled": cancelled,
        "provider": resolved["provider"],
        "model": conversation.get("model") or resolved["model"],
        "user_message": user_message,
        "assistant_message": assistant_message,
        "rag": [
            {"source": os.path.basename(r.get("source") or ""), "score": r.get("score"), "mode": r.get("mode")}
            for r in rag_results
        ],
        "messages_sent": len(messages),
        "seconds": round(time.time() - started, 2),
    }


def regenerate(
    conversation_id: str,
    message_id: str | None = None,
    *,
    on_delta=None,
    on_reasoning=None,
    **overrides,
) -> dict:
    """Drop an assistant answer (and everything after) and produce a new one."""
    conversation = resolve_conversation(conversation_id, **overrides)
    messages = store.list_messages(conversation_id)
    target = None
    if message_id:
        target = next((message for message in messages if message["id"] == message_id), None)
    if target is None:
        for message in reversed(messages):
            if message["role"] == "assistant":
                target = message
                break
    if target is None:
        raise ValueError("No reply to regenerate in this conversation.")

    index = messages.index(target)
    last_user = None
    for message in reversed(messages[:index]):
        if message["role"] == "user":
            last_user = message
            break
    if last_user is None:
        raise ValueError("No user message before this reply.")

    store.delete_from_message(conversation_id, last_user["id"], include=True)
    return run_chat(
        conversation_id,
        last_user.get("content") or "",
        images=list(last_user.get("images") or []),
        on_delta=on_delta,
        on_reasoning=on_reasoning,
        **overrides,
    )


def edit_message(conversation_id: str, message_id: str, content: str, images=None) -> dict:
    """Edit a message; everything after it is removed (like ChatGPT's edit)."""
    message = store.get_message(message_id)
    if not message or message["conversation_id"] != conversation_id:
        raise ValueError("Message not found.")
    store.update_message(message_id, content, images)
    store.delete_from_message(conversation_id, message_id, include=False)
    return store.get_conversation(conversation_id) or {}
