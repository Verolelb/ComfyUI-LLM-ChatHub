"""HTTP API for the ChatHub sidebar panel.

Everything is mounted under ``/llm_chathub``. Long running work (LLM calls, folder
indexing) happens in a worker thread and pushes events into an asyncio queue so the
browser receives server-sent events as they are produced.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import json
import logging
import os
import re
import threading

from aiohttp import web

from . import config, engine, providers, rag, store

LOGGER = logging.getLogger("llm_chathub")

_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_DOC_NAME = re.compile(r"[^A-Za-z0-9 ._-]+")


def _document_name(raw: str) -> str:
    """Reduce an uploaded file name to a safe base name (no path parts)."""
    name = os.path.basename((raw or "").replace("\\", "/")).strip()
    name = _DOC_NAME.sub("_", name).strip(" ._-")
    return name[:120]


def _document_rows(folder: str) -> list[dict]:
    rows = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        rows.append({"name": name, "size": os.path.getsize(path), "modified": int(os.path.getmtime(path))})
    return rows


def _json_error(message: str, status: int = 400) -> web.Response:
    return web.json_response({"ok": False, "error": message}, status=status)


def _safe(handler):
    """Never leak a traceback to the browser: unexpected errors become JSON 500."""

    @functools.wraps(handler)
    async def wrapper(request: web.Request):
        try:
            return await handler(request)
        except web.HTTPException:
            raise
        except ValueError as error:
            return _json_error(str(error), 409)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("llm_chathub %s failed: %s", request.path, error)
            return _json_error(f"Internal error: {error}", 500)

    return wrapper


async def _payload(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    try:
        data = await request.json()
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _public_message(message: dict) -> dict:
    images = [
        {"name": name, "url": f"/llm_chathub/image/{name}"}
        for name in (message.get("images") or [])
    ]
    meta = message.get("meta") or {}
    return {
        "id": message.get("id"),
        "role": message.get("role"),
        "content": message.get("content") or "",
        "images": images,
        "created_at": message.get("created_at"),
        "edited": bool(message.get("edited")),
        "meta": meta,
    }


def _public_conversation(conversation: dict) -> dict:
    return {
        "id": conversation.get("id"),
        "title": conversation.get("title"),
        "provider": conversation.get("provider"),
        "model": conversation.get("model"),
        "system_prompt": conversation.get("system_prompt"),
        "reference": conversation.get("reference"),
        "rag_enabled": bool(conversation.get("rag_enabled")),
        "rag_folder": conversation.get("rag_folder"),
        "rag_top_k": conversation.get("rag_top_k"),
        "history_limit": conversation.get("history_limit"),
        "temperature": conversation.get("temperature"),
        "max_tokens": conversation.get("max_tokens"),
        "created_at": conversation.get("created_at"),
        "updated_at": conversation.get("updated_at"),
        "message_count": conversation.get("message_count", 0),
    }


# --------------------------------------------------------------------------- streaming


async def _stream_worker(request: web.Request, worker, *, conversation_id: str | None = None):
    """Run a blocking job in a thread while streaming its events over SSE.

    ``worker`` receives one ``emit(kind, payload)`` callable and returns the final
    result, which is sent as a ``result`` event before the stream closes.
    """
    stream = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await stream.prepare(request)

    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()

    def emit(kind: str, payload) -> None:
        try:
            loop.call_soon_threadsafe(events.put_nowait, (kind, payload))
        except RuntimeError:
            pass

    def runner() -> None:
        try:
            result = worker(emit)
            emit("result", result)
        except Exception as error:  # noqa: BLE001 - surfaced to the UI as text
            LOGGER.warning("llm_chathub worker failed: %s", error)
            emit("error", str(error))
        finally:
            emit("end", None)

    threading.Thread(target=runner, daemon=True).start()

    async def send(kind: str, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False)
        await stream.write(f"event: {kind}\ndata: {body}\n\n".encode("utf-8"))

    try:
        while True:
            kind, payload = await events.get()
            if kind == "end":
                break
            await send(kind, payload)
    except (ConnectionResetError, asyncio.CancelledError):
        if conversation_id:
            engine.request_cancel(conversation_id)
    finally:
        try:
            await stream.write_eof()
        except Exception:
            pass
    return stream


# --------------------------------------------------------------------------- routes


def register_routes() -> None:
    try:
        from server import PromptServer  # type: ignore

        routes = PromptServer.instance.routes
    except Exception as error:  # pragma: no cover - only outside ComfyUI
        LOGGER.warning("llm_chathub: API routes unavailable (%s)", error)
        return

    # ---------------------------------------------------------------- state

    @routes.get("/llm_chathub/state")
    async def state(request: web.Request):
        settings = config.public_settings()
        return web.json_response(
            {
                "ok": True,
                "settings": settings,
                "conversations": [
                    _public_conversation(item) for item in store.list_conversations()
                ],
                "index": store.index_stats(),
                "version": "1.0",
            }
        )

    @routes.post("/llm_chathub/settings")
    async def save_settings(request: web.Request):
        data = await _payload(request)
        settings = config.update_settings(data)
        return web.json_response({"ok": True, "settings": config.public_settings()})

    @routes.post("/llm_chathub/models")
    async def list_models(request: web.Request):
        data = await _payload(request)
        try:
            resolved = config.resolve_provider(
                data.get("provider"),
                model=None,
                base_url=data.get("base_url") or None,
                api_key=data.get("api_key") if data.get("api_key") else None,
            )
            names = await asyncio.to_thread(
                providers.list_models, resolved["provider"], resolved["base_url"], resolved["api_key"]
            )
        except Exception as error:  # noqa: BLE001
            return _json_error(str(error), 502)
        return web.json_response({"ok": True, "models": names})

    # ---------------------------------------------------------------- conversations

    @routes.get("/llm_chathub/conversations")
    async def conversations(request: web.Request):
        return web.json_response(
            {
                "ok": True,
                "conversations": [
                    _public_conversation(item) for item in store.list_conversations()
                ],
            }
        )

    @routes.get("/llm_chathub/conversation")
    async def get_conversation(request: web.Request):
        conversation_id = request.query.get("id") or ""
        conversation = store.get_conversation(conversation_id)
        if not conversation:
            return _json_error("Conversation not found.", 404)
        messages = store.list_messages(conversation_id)
        return web.json_response(
            {
                "ok": True,
                "conversation": _public_conversation(conversation),
                "messages": [_public_message(message) for message in messages],
            }
        )

    @routes.post("/llm_chathub/conversation/new")
    @_safe
    async def new_conversation(request: web.Request):
        data = await _payload(request)
        fields = {key: data[key] for key in data if key in store.CONVERSATION_FIELDS}
        conversation_id = data.get("id") or None
        if conversation_id and store.get_conversation(conversation_id):
            raise ValueError(f"Conversation {conversation_id} already exists.")
        conversation = store.create_conversation(id=conversation_id, **fields)
        config.update_settings({"last_conversation": conversation["id"]})
        return web.json_response({"ok": True, "conversation": _public_conversation(conversation)})

    @routes.post("/llm_chathub/conversation/update")
    @_safe
    async def update_conversation(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("id")
        if not conversation_id:
            return _json_error("id manquant.")
        fields = {key: data[key] for key in data if key in store.CONVERSATION_FIELDS}
        conversation = store.update_conversation(conversation_id, **fields)
        if not conversation:
            return _json_error("Conversation not found.", 404)
        return web.json_response({"ok": True, "conversation": _public_conversation(conversation)})

    @routes.post("/llm_chathub/conversation/delete")
    async def delete_conversation(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("id")
        if not conversation_id:
            return _json_error("id manquant.")
        store.delete_conversation(conversation_id)
        return web.json_response({"ok": True})

    # ---------------------------------------------------------------- chat

    @routes.post("/llm_chathub/chat")
    async def chat(request: web.Request):
        data = await _payload(request)
        text = (data.get("text") or "").strip()
        conversation_id = data.get("conversation_id") or None
        images = data.get("images") or []
        if not text and not images:
            return _json_error("Empty message.")
        forward = data.get("overrides") or {}

        def worker(emit):
            return engine.run_chat(
                conversation_id,
                text,
                image_payloads=images,
                on_delta=lambda delta: emit("delta", delta),
                on_reasoning=lambda text: emit("reasoning", text),
                **forward,
            )

        return await _stream_worker(request, worker, conversation_id=conversation_id)

    @routes.post("/llm_chathub/regenerate")
    async def regenerate(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("conversation_id")
        message_id = data.get("message_id")
        if not conversation_id:
            return _json_error("conversation_id is required.")

        def worker(emit):
            return engine.regenerate(
                conversation_id,
                message_id,
                on_delta=lambda delta: emit("delta", delta),
                on_reasoning=lambda text: emit("reasoning", text),
                **(data.get("overrides") or {}),
            )

        return await _stream_worker(request, worker, conversation_id=conversation_id)

    @routes.post("/llm_chathub/cancel")
    async def cancel(request: web.Request):
        data = await _payload(request)
        return web.json_response(
            {"ok": True, "cancelled": engine.request_cancel(data.get("conversation_id") or "")}
        )

    @routes.post("/llm_chathub/messages/edit")
    @_safe
    async def edit_message(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("conversation_id")
        message_id = data.get("message_id")
        if not conversation_id or not message_id:
            return _json_error("conversation_id and message_id are required.")
        try:
            engine.edit_message(conversation_id, message_id, data.get("content") or "", data.get("images"))
        except Exception as error:  # noqa: BLE001
            return _json_error(str(error))
        return web.json_response(
            {
                "ok": True,
                "messages": [_public_message(m) for m in store.list_messages(conversation_id)],
            }
        )

    @routes.post("/llm_chathub/messages/delete")
    async def delete_message(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("conversation_id")
        message_id = data.get("message_id")
        if not message_id or not conversation_id:
            return _json_error("conversation_id and message_id are required.")
        if data.get("from_here"):
            store.delete_from_message(conversation_id, message_id, include=True)
        else:
            store.delete_message(message_id)
        return web.json_response(
            {
                "ok": True,
                "messages": [_public_message(m) for m in store.list_messages(conversation_id)],
            }
        )

    @routes.post("/llm_chathub/messages/truncate")
    async def truncate_messages(request: web.Request):
        data = await _payload(request)
        conversation_id = data.get("conversation_id")
        if not conversation_id:
            return _json_error("conversation_id is required.")
        store.truncate_messages(conversation_id, int(data.get("keep") or 0))
        return web.json_response(
            {
                "ok": True,
                "messages": [_public_message(m) for m in store.list_messages(conversation_id)],
            }
        )

    # ---------------------------------------------------------------- images

    @routes.post("/llm_chathub/upload")
    @_safe
    async def upload(request: web.Request):
        data = await _payload(request)
        name = (data.get("name") or "image.png").strip() or "image.png"
        suffix = os.path.splitext(name)[1].lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"):
            suffix = ".png"
        payload = data.get("data") or ""
        if not payload:
            return _json_error("Empty image.")
        try:
            stored = engine.save_base64_image(payload, suffix)
        except Exception as error:  # noqa: BLE001
            return _json_error(f"Could not save the image: {error}")
        return web.json_response({"ok": True, "name": stored, "url": f"/llm_chathub/image/{stored}"})

    @routes.get("/llm_chathub/image/{name}")
    async def image(request: web.Request):
        name = request.match_info.get("name", "")
        if not _SAFE_NAME.match(name):
            return _json_error("Invalid name.", 400)
        path = os.path.join(config.images_dir(), name)
        if not os.path.isfile(path):
            return _json_error("Image not found.", 404)
        response = web.FileResponse(path)
        response.headers["Cache-Control"] = "public, max-age=604800"
        return response

    # ---------------------------------------------------------------- rag

    @routes.get("/llm_chathub/rag/files")
    async def rag_files(request: web.Request):
        folder = config.documents_dir()
        os.makedirs(folder, exist_ok=True)
        return web.json_response({"ok": True, "folder": folder, "files": _document_rows(folder)})

    @routes.post("/llm_chathub/rag/files")
    @_safe
    async def rag_files_upload(request: web.Request):
        data = await _payload(request)
        files = data.get("files")
        if not isinstance(files, list) or not files:
            return _json_error("No file received.")
        folder = config.documents_dir()
        os.makedirs(folder, exist_ok=True)
        max_bytes = int(float(data.get("max_file_mb") or 8.0) * 1024 * 1024)
        saved, skipped = [], []
        for item in files:
            if not isinstance(item, dict):
                continue
            name = _document_name(str(item.get("name") or ""))
            if not name:
                skipped.append("(invalid name)")
                continue
            if os.path.splitext(name)[1].lower() not in rag.TEXT_EXTENSIONS:
                skipped.append(f"{name} (format not read by the RAG)")
                continue
            payload = str(item.get("data") or "")
            if not payload:
                skipped.append(f"{name} (empty)")
                continue
            try:
                raw = base64.b64decode(payload.split(",", 1)[1] if payload.startswith("data:") else payload)
            except Exception:
                skipped.append(f"{name} (unreadable content)")
                continue
            if len(raw) > max_bytes:
                skipped.append(f"{name} (over {max_bytes // (1024 * 1024)} MB)")
                continue
            with open(os.path.join(folder, name), "wb") as handle:
                handle.write(raw)
            saved.append(name)
        if not saved:
            return web.json_response(
                {"ok": False, "error": "No file saved.", "skipped": skipped, "folder": folder},
                status=400,
            )
        return web.json_response(
            {"ok": True, "folder": folder, "saved": saved, "skipped": skipped, "files": _document_rows(folder)}
        )

    @routes.post("/llm_chathub/rag/files/delete")
    @_safe
    async def rag_files_delete(request: web.Request):
        data = await _payload(request)
        name = _document_name(str(data.get("name") or ""))
        if not name:
            return _json_error("Missing name.")
        path = os.path.join(config.documents_dir(), name)
        if os.path.isfile(path):
            os.remove(path)
        return web.json_response(
            {"ok": True, "folder": config.documents_dir(), "files": _document_rows(config.documents_dir())}
        )

    @routes.get("/llm_chathub/rag/status")
    async def rag_status(request: web.Request):
        return web.json_response(
            {
                "ok": True,
                "index": store.index_stats(),
                "documents": store.list_documents(),
            }
        )

    @routes.post("/llm_chathub/rag/index")
    @_safe
    async def rag_index(request: web.Request):
        data = await _payload(request)
        folder = (data.get("folder") or "").strip() or config.documents_dir()
        if not os.path.isdir(folder):
            return _json_error(f"Folder not found: {folder}")
        resolved = config.resolve_provider(data.get("provider"))
        options = {
            "provider": resolved["provider"],
            "base_url": data.get("base_url") or resolved["base_url"],
            "api_key": resolved["api_key"],
            "embed_model": (data.get("embed_model") or resolved["embed_model"] or "").strip(),
            "chunk_size": int(data.get("chunk_size") or resolved["rag_chunk_size"]),
            "overlap": int(data.get("overlap") or resolved["rag_overlap"]),
            "max_file_mb": float(data.get("max_file_mb") or resolved["rag_max_file_mb"]),
            "replace": bool(data.get("replace", True)),
        }

        def worker(emit):
            return rag.index_folder(
                folder,
                progress=lambda stage, done, total, message: emit(
                    "progress", {"stage": stage, "done": done, "total": total, "message": message}
                ),
                **options,
            )

        return await _stream_worker(request, worker)

    @routes.post("/llm_chathub/rag/clear")
    async def rag_clear(request: web.Request):
        store.clear_index()
        return web.json_response({"ok": True, "index": store.index_stats()})

    @routes.post("/llm_chathub/rag/test")
    async def rag_test(request: web.Request):
        data = await _payload(request)
        resolved = config.resolve_provider(data.get("provider"))
        results = rag.retrieve(
            data.get("query") or "",
            top_k=int(data.get("top_k") or resolved["rag_top_k"]),
            provider=resolved["provider"],
            base_url=resolved["base_url"],
            api_key=resolved["api_key"],
            embed_model=(data.get("embed_model") or resolved["embed_model"] or "").strip(),
        )
        return web.json_response({"ok": True, "results": results})

    # ---------------------------------------------------------------- diagnostics

    @routes.get("/llm_chathub/health")
    async def health(request: web.Request):
        return web.json_response(
            {
                "ok": True,
                "base_dir": config.base_dir(),
                "images_dir": config.images_dir(),
                "providers": list(config.PROVIDER_DEFAULTS),
            }
        )

    LOGGER.info("llm_chathub: API routes registered under /llm_chathub")
