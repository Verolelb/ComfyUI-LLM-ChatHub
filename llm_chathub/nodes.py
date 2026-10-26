"""ComfyUI nodes for LLM ChatHub.

The nodes share the same conversation store as the sidebar panel, so a session
started in the graph can be continued in the panel and the other way round.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

from . import config, engine, rag, store

CATEGORY = "LLM ChatHub"
CONFIG_TYPE = "CHATHUB_CONFIG"
SESSION_TYPE = "CHATHUB_SESSION"

USE_PANEL = "panel (saved settings)"
# The panel entry and the combo values below used to be French; workflows saved back
# then still carry those strings, so every alias is accepted on the way in.
PANEL_CHOICES = {USE_PANEL, "panel (reglages enregistres)", "panel"}
PROVIDER_CHOICES = [USE_PANEL, "lmstudio", "ollama", "ollama_cloud"]

RESET_CHOICES = ["none", "clear history", "resume from system prompt"]
RESET_ALIASES = {
    "aucun": "none",
    "effacer l'historique": "clear history",
    "reprendre avec le prompt systeme": "resume from system prompt",
}

IMAGE_MODE_CHOICES = ["first image", "every image in the batch"]
IMAGE_MODE_ALIASES = {
    "premiere image": "first image",
    "chaque image du lot": "every image in the batch",
}

HISTORY_OPERATIONS = [
    "nothing",
    "clear all",
    "delete the last N messages",
    "truncate to N messages",
    "set the system prompt",
]
HISTORY_ALIASES = {
    "rien": "nothing",
    "effacer tout": "clear all",
    "supprimer les N derniers messages": "delete the last N messages",
    "tronquer a N messages": "truncate to N messages",
    "definir le prompt systeme": "set the system prompt",
}


def _upgrade(value: str, aliases: dict) -> str:
    """Map a widget value saved before the translation onto its English name."""
    return aliases.get((value or "").strip(), value)


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value or "").strip("_")
    return cleaned[:40]


def _make_config(
    provider_choice: str,
    base_url: str = "",
    model: str = "",
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    history_limit: int = 20,
    embed_model: str = "",
) -> dict:
    panel = provider_choice in PANEL_CHOICES
    name = None if panel else provider_choice
    # With the panel choice the connection is the panel's, not the leftovers of this
    # node's widgets: they keep the URL/model of a previous provider, and honouring
    # them sent an "Ollama Cloud" conversation to the LM Studio port.
    resolved = config.resolve_provider(
        name,
        model=None if panel else ((model or "").strip() or None),
        base_url=None if panel else ((base_url or "").strip() or None),
        api_key=None if panel else ((api_key or "").strip() or None),
    )
    return {
        "provider": resolved["provider"],
        "label": resolved["label"],
        "base_url": resolved["base_url"],
        "model": resolved["model"],
        "api_key": resolved["api_key"],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "history_limit": int(history_limit),
        "embed_model": (embed_model or "").strip() or resolved["embed_model"],
        "timeout": resolved["timeout"],
        "source": provider_choice,
    }


def _transcript(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = {"user": "User", "assistant": "Assistant", "system": "System"}.get(
            message.get("role"), message.get("role")
        )
        images = message.get("images") or []
        suffix = f"  [{len(images)} image(s)]" if images else ""
        lines.append(f"### {role}{suffix}\n{message.get('content') or ''}".rstrip())
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- nodes


class LCProvider:
    """Pick the LLM backend and connection parameters."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (PROVIDER_CHOICES, {"default": USE_PANEL}),
                "base_url": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 131072}),
                "history_limit": ("INT", {"default": 20, "min": 0, "max": 200}),
                "embed_model": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = (CONFIG_TYPE, "STRING")
    RETURN_NAMES = ("config", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # The connection lives in the panel, outside the graph: without this ComfyUI
        # reuses the cached config and a provider switch never reaches the nodes.
        return config.connection_fingerprint()

    def run(self, provider, base_url, model, api_key, temperature, max_tokens, history_limit, embed_model):
        conf = _make_config(
            provider, base_url, model, api_key, temperature, max_tokens, history_limit, embed_model
        )
        info = (
            f"{conf['label']} | url={conf['base_url']} | model={conf['model'] or '(none)'} "
            f"| key={'yes' if conf['api_key'] else 'no'}"
        )
        return (conf, info)


class LCSession:
    """Open (or reset) a persistent conversation and set the starting prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (CONFIG_TYPE,),
                "session_id": ("STRING", {"default": "auto", "multiline": False}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "reference_text": ("STRING", {"default": "", "multiline": True}),
                "history_limit": ("INT", {"default": 0, "min": 0, "max": 200}),
                "reset": (RESET_CHOICES,),
                "rag_enabled": ("BOOLEAN", {"default": False}),
                "rag_folder": ("STRING", {"default": "", "multiline": False}),
                "rag_top_k": ("INT", {"default": 4, "min": 1, "max": 20}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (SESSION_TYPE, "STRING")
    RETURN_NAMES = ("session", "transcript")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(
        self,
        config,
        session_id,
        system_prompt,
        reference_text,
        history_limit,
        reset,
        rag_enabled,
        rag_folder,
        rag_top_k,
        unique_id=None,
    ):
        reset = _upgrade(reset, RESET_ALIASES)
        raw = (session_id or "").strip()
        if not raw or raw.lower() == "auto":
            conversation_id = f"conv_node_{_slug(str(unique_id or '0'))}"
        elif raw.startswith("conv_"):
            conversation_id = raw
        else:
            conversation_id = f"conv_{_slug(raw)}"

        fields = {
            "provider": config.get("provider") or None,
            "model": config.get("model") or None,
            "system_prompt": (system_prompt or "").strip() or None,
            "reference": reference_text or "",
            "rag_enabled": bool(rag_enabled),
            "rag_folder": rag_folder or "",
            "rag_top_k": int(rag_top_k),
            "history_limit": int(history_limit or config.get("history_limit") or 0),
            "temperature": float(config.get("temperature", 0.7)),
            "max_tokens": int(config.get("max_tokens", 1024)),
        }
        fields = {key: value for key, value in fields.items() if value is not None}

        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            conversation = store.create_conversation(id=conversation_id, **fields)
        elif fields:
            conversation = store.update_conversation(conversation_id, **fields) or conversation

        if reset == "clear history":
            store.truncate_messages(conversation_id, 0)
        elif reset == "resume from system prompt":
            store.truncate_messages(conversation_id, 1)

        messages = store.list_messages(conversation_id)
        session = {
            "conversation_id": conversation_id,
            "config": config,
            "history_limit": int(history_limit or 0),
            "rag": {"enabled": bool(rag_enabled), "folder": rag_folder, "top_k": int(rag_top_k)},
        }
        return (session, _transcript(messages))


class LCChat:
    """Send a message to the model and get the reply back as text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "message": ("STRING", {"default": "", "multiline": True}),
                "image_mode": (IMAGE_MODE_CHOICES,),
                "persist": ("BOOLEAN", {"default": True}),
                "context": ("STRING", {"default": "", "multiline": True}),
                # ``control_after_generate`` gives the usual fixed / randomize /
                # increment / decrement control under the value, from either node
                # definition (v1 INPUT_TYPES here, v3 ``io.Int`` for new nodes).
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    # "fixed" instead of True: the value is the control's default mode,
                    # so a chat node keeps answering the same until you ask otherwise.
                    "control_after_generate": "fixed",
                    "tooltip": "Sampling seed sent to the model. Fixed reuses the cached answer, randomize/increment asks again.",
                }),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("reply", "transcript", "conversation_id")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, session, message, image_mode, persist, context="", seed=0, image=None):
        conversation_id = session.get("conversation_id")
        conf = session.get("config") or {}
        connection = {
            "base_url": conf.get("base_url"),
            "api_key": conf.get("api_key"),
            "embed_model": conf.get("embed_model"),
            "timeout": conf.get("timeout"),
        }
        overrides = {
            "provider": conf.get("provider"),
            "model": conf.get("model"),
            "temperature": conf.get("temperature"),
            "max_tokens": conf.get("max_tokens"),
        }
        if session.get("history_limit"):
            overrides["history_limit"] = session["history_limit"]

        image_mode = _upgrade(image_mode, IMAGE_MODE_ALIASES)
        text = (message or "").strip()
        if context.strip():
            text = f"--- CONTEXT ---\n{context.strip()}\n--- END OF CONTEXT ---\n\n{text}"

        frames = engine.split_image_batch(image) if image is not None else [None]
        if image_mode == "first image":
            frames = frames[:1]

        replies = []
        for frame in frames:
            payloads = [engine.tensor_to_base64(frame)] if frame is not None else []
            result = engine.run_chat(
                conversation_id,
                text,
                image_payloads=payloads if persist else None,
                attach_b64=None if persist else payloads,
                persist=bool(persist),
                connection=connection,
                seed=None if seed is None else int(seed),
                **overrides,
            )
            if result.get("error") and not result.get("reply"):
                raise RuntimeError(result["error"])
            if not result.get("reply") and result.get("reasoning"):
                raise RuntimeError(
                    "The model reasoned without producing a reply: raise 'max_tokens' "
                    "(the reasoning consumes the budget)."
                )
            replies.append(result.get("reply") or "")

        messages = store.list_messages(conversation_id) if persist else []
        return ("\n\n".join(replies), _transcript(messages), conversation_id or "")


class LCRagIndex:
    """Index a folder of text files for retrieval."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (CONFIG_TYPE,),
                "folder": ("STRING", {"default": "", "multiline": False}),
                "embed_model": ("STRING", {"default": "", "multiline": False}),
                "chunk_size": ("INT", {"default": 1200, "min": 200, "max": 8000}),
                "overlap": ("INT", {"default": 200, "min": 0, "max": 2000}),
                "replace": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("report", "chunks", "documents")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, config, folder, embed_model, chunk_size, overlap, replace):
        blob = json.dumps([config.get("provider"), folder, embed_model, chunk_size, overlap, replace])
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def run(self, config, folder, embed_model, chunk_size, overlap, replace):
        summary = rag.index_folder(
            folder,
            provider=config.get("provider") or "lmstudio",
            base_url=config.get("base_url") or "",
            api_key=config.get("api_key") or "",
            embed_model=(embed_model or "").strip() or config.get("embed_model") or "",
            chunk_size=int(chunk_size),
            overlap=int(overlap),
            replace=bool(replace),
        )
        lines = [
            f"Folder: {summary['folder']}",
            f"Documents indexed: {summary['documents']}",
            f"Excerpts: {summary['chunks']}",
            f"Embeddings: {'yes' if summary['embedded'] else 'no (keyword fallback)'}",
        ]
        if summary.get("embedding_warning"):
            lines.append(f"Embedding warning: {summary['embedding_warning']}")
        if summary.get("skipped"):
            lines.append(f"Skipped: {', '.join(summary['skipped'][:10])}")
        return ("\n".join(lines), int(summary["chunks"]), int(summary["documents"]))


class LCRagRetrieve:
    """Retrieve the most relevant chunks for a query."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (CONFIG_TYPE,),
                "query": ("STRING", {"default": "", "multiline": True}),
                "top_k": ("INT", {"default": 4, "min": 1, "max": 30}),
                "max_chars": ("INT", {"default": 6000, "min": 500, "max": 40000}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("context", "sources")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, config, query, top_k, max_chars):
        results = rag.retrieve(
            query,
            top_k=int(top_k),
            provider=config.get("provider") or "lmstudio",
            base_url=config.get("base_url") or "",
            api_key=config.get("api_key") or "",
            embed_model=config.get("embed_model") or "",
        )
        context = rag.build_context(results, max_chars=int(max_chars))
        sources = "\n".join(
            f"{os.path.basename(result.get('source') or '')} (score {result.get('score')}, {result.get('mode')})"
            for result in results
        )
        return (context, sources)


class LCHistory:
    """Read the stored conversation back into the graph."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "limit": ("INT", {"default": 0, "min": 0, "max": 500}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("transcript", "last_reply", "count")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, session, limit):
        messages = store.list_messages(session.get("conversation_id") or "")
        messages = [message for message in messages if message.get("role") != "system"]
        if limit:
            messages = messages[-int(limit):]
        last_reply = ""
        for message in reversed(messages):
            if message.get("role") == "assistant":
                last_reply = message.get("content") or ""
                break
        return (_transcript(messages), last_reply, len(messages))


class LCHistoryEdit:
    """Clear or trim the stored history of a session."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "session": (SESSION_TYPE,),
                "operation": (HISTORY_OPERATIONS,),
                "count": ("INT", {"default": 1, "min": 0, "max": 500}),
                "new_system_prompt": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("transcript", "count")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, session, operation, count, new_system_prompt):
        operation = _upgrade(operation, HISTORY_ALIASES)
        conversation_id = session.get("conversation_id") or ""
        messages = store.list_messages(conversation_id)

        if operation == "clear all" and messages:
            store.truncate_messages(conversation_id, 0)
        elif operation == "delete the last N messages" and messages:
            keep = max(0, len(messages) - int(count))
            store.truncate_messages(conversation_id, keep)
        elif operation == "truncate to N messages":
            store.truncate_messages(conversation_id, int(count))
        elif operation == "set the system prompt":
            store.update_conversation(conversation_id, system_prompt=new_system_prompt or "")

        messages = store.list_messages(conversation_id)
        return (_transcript(messages), len(messages))


NODE_CLASS_MAPPINGS = {
    "LLMChatHubProvider": LCProvider,
    "LLMChatHubSession": LCSession,
    "LLMChatHubChat": LCChat,
    "LLMChatHubRagIndex": LCRagIndex,
    "LLMChatHubRagRetrieve": LCRagRetrieve,
    "LLMChatHubHistory": LCHistory,
    "LLMChatHubHistoryEdit": LCHistoryEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMChatHubProvider": "LLM ChatHub - Provider (LM Studio / Ollama)",
    "LLMChatHubSession": "LLM ChatHub - Session + starting prompt",
    "LLMChatHubChat": "LLM ChatHub - Chat",
    "LLMChatHubRagIndex": "LLM ChatHub - RAG Indexer",
    "LLMChatHubRagRetrieve": "LLM ChatHub - RAG Retrieve",
    "LLMChatHubHistory": "LLM ChatHub - History (read)",
    "LLMChatHubHistoryEdit": "LLM ChatHub - History (edit)",
}
