"""Filesystem paths and persisted settings for the LLM ChatHub node pack.

Everything user specific (database, uploaded images, downloaded index) lives in
ComfyUI's own ``user`` directory so that updating or reinstalling the node pack
never destroys a chat history.
"""

from __future__ import annotations

import copy
import json
import os
import threading

# custom_nodes/ComfyUI-LLM-ChatHub/llm_chathub/config.py -> custom_nodes/ComfyUI-LLM-ChatHub
NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# -> ComfyUI/
COMFY_ROOT = os.path.dirname(os.path.dirname(NODE_DIR))

_LOCK = threading.RLock()

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, precise and concise assistant. "
    "You answer in the user's language and follow their instructions."
)

# --------------------------------------------------------------------------- providers

PROVIDER_DEFAULTS = {
    "lmstudio": {
        "label": "LM Studio (local)",
        "base_url": "http://127.0.0.1:1234/v1",
        "needs_key": False,
        "hint": "In LM Studio: Developer tab -> Start Server.",
    },
    "ollama": {
        "label": "Ollama (local)",
        "base_url": "http://127.0.0.1:11434",
        "needs_key": False,
        "hint": "Run `ollama serve`, then `ollama pull <model>`.",
    },
    "ollama_cloud": {
        "label": "Ollama Cloud",
        "base_url": "https://ollama.com",
        "needs_key": True,
        "hint": "API key at https://ollama.com/settings/keys",
    },
}

PROVIDER_ALIASES = {
    "lm studio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "local": "lmstudio",
    "ollama cloud": "ollama_cloud",
    "ollama-cloud": "ollama_cloud",
    "cloud": "ollama_cloud",
}

DEFAULT_SETTINGS = {
    "active_provider": "lmstudio",
    "providers": {
        "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "model": "", "api_key": ""},
        "ollama": {"base_url": "http://127.0.0.1:11434", "model": "", "api_key": ""},
        "ollama_cloud": {"base_url": "https://ollama.com", "model": "", "api_key": ""},
    },
    "chat": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "timeout": 600,
        "history_limit": 20,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
    },
    "rag": {
        "enabled": False,
        "folder": "",
        "top_k": 4,
        "chunk_size": 1200,
        "overlap": 200,
        "embed_model": "",
        "max_file_mb": 8,
    },
    "last_conversation": "",
}

ENV_KEYS = {
    "lmstudio": ("LMSTUDIO_API_KEY", "LM_API_TOKEN"),
    "ollama": ("OLLAMA_API_KEY",),
    "ollama_cloud": ("OLLAMA_API_KEY", "OLLAMA_CLOUD_API_KEY"),
}


def normalize_provider(value) -> str:
    """Accept loose spellings ('LM Studio', 'ollama-cloud', ...)."""
    if not value:
        return "lmstudio"
    key = str(value).strip().lower()
    key = PROVIDER_ALIASES.get(key, key.replace("-", "_").replace(" ", "_"))
    if key not in PROVIDER_DEFAULTS:
        raise ValueError(
            f"Provider inconnu: {value!r}. Valeurs acceptees: "
            + ", ".join(sorted(PROVIDER_DEFAULTS))
        )
    return key


def provider_label(value) -> str:
    try:
        return PROVIDER_DEFAULTS[normalize_provider(value)]["label"]
    except Exception:
        return str(value)


# --------------------------------------------------------------------------- paths


def _user_directory() -> str:
    override = os.environ.get("LLM_CHATHUB_HOME")
    if override:
        return override
    try:
        import folder_paths  # type: ignore

        directory = folder_paths.get_user_directory()
        if directory:
            return directory
    except Exception:
        pass
    return os.path.join(COMFY_ROOT, "user")


def base_dir() -> str:
    return os.path.join(_user_directory(), "llm_chathub")


def db_path() -> str:
    return os.path.join(base_dir(), "llm_chathub.sqlite3")


def images_dir() -> str:
    return os.path.join(base_dir(), "images")


def documents_dir() -> str:
    return os.path.join(base_dir(), "documents")


def settings_path() -> str:
    return os.path.join(base_dir(), "settings.json")


def ensure_dirs() -> None:
    for path in (base_dir(), images_dir(), documents_dir()):
        os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------- settings


def _deep_merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _env_key(provider: str) -> str:
    for name in ENV_KEYS.get(provider, ()):
        value = os.environ.get(name)
        if value:
            return value.strip()
    return ""


def load_settings() -> dict:
    """Read settings.json, always returning a complete settings structure."""
    with _LOCK:
        ensure_dirs()
        data = {}
        path = settings_path()
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = {}
        merged = _deep_merge(DEFAULT_SETTINGS, data if isinstance(data, dict) else {})
        for provider in PROVIDER_DEFAULTS:
            merged["providers"].setdefault(provider, {})
            merged["providers"][provider].setdefault(
                "base_url", PROVIDER_DEFAULTS[provider]["base_url"]
            )
            merged["providers"][provider].setdefault("model", "")
            merged["providers"][provider].setdefault("api_key", "")
        try:
            merged["active_provider"] = normalize_provider(merged.get("active_provider"))
        except ValueError:
            merged["active_provider"] = "lmstudio"
        return merged


def save_settings(settings: dict) -> dict:
    with _LOCK:
        ensure_dirs()
        merged = _deep_merge(DEFAULT_SETTINGS, settings or {})
        tmp = settings_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(merged, handle, ensure_ascii=False, indent=2)
        os.replace(tmp, settings_path())
        return merged


def update_settings(patch: dict) -> dict:
    return save_settings(_deep_merge(load_settings(), patch or {}))


def resolve_provider(provider=None, model=None, base_url=None, api_key=None) -> dict:
    """Merge stored settings, explicit overrides and environment variables."""
    settings = load_settings()
    name = normalize_provider(provider or settings.get("active_provider"))
    stored = settings["providers"].get(name, {})
    chat = settings.get("chat", {})
    resolved = {
        "provider": name,
        "label": PROVIDER_DEFAULTS[name]["label"],
        "base_url": (base_url or stored.get("base_url") or PROVIDER_DEFAULTS[name]["base_url"]).rstrip("/"),
        "model": (model or stored.get("model") or "").strip(),
        "api_key": (api_key if api_key is not None and api_key != "" else (stored.get("api_key") or _env_key(name))),
        "temperature": float(chat.get("temperature", 0.7)),
        "max_tokens": int(chat.get("max_tokens", 1024)),
        "timeout": float(chat.get("timeout", 600)),
        "history_limit": int(chat.get("history_limit", 20)),
        "system_prompt": chat.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
    }
    rag = settings.get("rag", {})
    resolved["rag_enabled"] = bool(rag.get("enabled", False))
    resolved["rag_folder"] = rag.get("folder", "") or ""
    resolved["rag_top_k"] = int(rag.get("top_k", 4))
    resolved["rag_chunk_size"] = int(rag.get("chunk_size", 1200))
    resolved["rag_overlap"] = int(rag.get("overlap", 200))
    resolved["embed_model"] = rag.get("embed_model", "") or ""
    resolved["rag_max_file_mb"] = float(rag.get("max_file_mb", 8))
    return resolved


def public_settings() -> dict:
    """Settings safe to send to the browser (api keys are masked)."""
    settings = load_settings()
    providers = {}
    for name, values in settings["providers"].items():
        providers[name] = {
            "base_url": values.get("base_url", PROVIDER_DEFAULTS[name]["base_url"]),
            "model": values.get("model", ""),
            "has_key": bool(values.get("api_key") or _env_key(name)),
        }
    return {
        "active_provider": settings.get("active_provider", "lmstudio"),
        "providers": providers,
        "chat": settings.get("chat", {}),
        "rag": settings.get("rag", {}),
        "last_conversation": settings.get("last_conversation", ""),
        "provider_defaults": PROVIDER_DEFAULTS,
        "default_system_prompt": DEFAULT_SYSTEM_PROMPT,
        "base_dir": base_dir(),
    }
