"""HTTP clients for LM Studio, Ollama and Ollama Cloud.

Chat and embeddings exist twice on purpose:

* a synchronous generator built on ``requests`` (used by the ComfyUI nodes, which
  already run on worker threads), and
* an asynchronous generator built on ``aiohttp`` (used by the browser panel so the
  reply can be streamed token by token).

Both share the same request builders and parsers, so the provider dialects live in
exactly one place.
"""

from __future__ import annotations

import json
import threading
from typing import Iterator

import requests

from . import config

USER_AGENT = "ComfyUI-LLM-ChatHub/1.0"


class ProviderError(RuntimeError):
    """Raised with a message already written for a human to read."""


def _base(base_url: str, fallback_provider: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        base = config.PROVIDER_DEFAULTS[fallback_provider]["base_url"]
    if fallback_provider in ("ollama", "ollama_cloud"):
        # Ollama speaks its own dialect at the root; a copied OpenAI-style base URL
        # (``https://ollama.com/v1``) would turn ``/api/tags`` into a 404.
        stripped = True
        while stripped:
            stripped = False
            for suffix in ("v1", "api"):
                if base.endswith("/" + suffix):
                    base = base[: -len(suffix) - 1].rstrip("/")
                    stripped = True
    return base


def _headers(api_key: str) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def endpoint(provider: str, base_url: str, path: str) -> str:
    return _base(base_url, provider) + path


def provider_hint(provider: str) -> str:
    return config.PROVIDER_DEFAULTS[provider]["hint"]


# --------------------------------------------------------------------------- payloads


def to_provider_messages(provider: str, messages: list[dict]) -> list[dict]:
    """Convert the internal message format to the provider dialect.

    Internal format: ``{"role": "user", "content": "...", "images": ["<b64 png>", ...]}``
    """
    if provider == "lmstudio":
        converted = []
        for message in messages:
            images = message.get("images") or []
            if not images:
                converted.append({"role": message["role"], "content": message.get("content", "")})
                continue
            content: list[dict] = []
            text = message.get("content") or ""
            if text:
                content.append({"type": "text", "text": text})
            for image in images:
                data_url = image if image.startswith("data:") else f"data:image/png;base64,{image}"
                content.append({"type": "image_url", "image_url": {"url": data_url}})
            converted.append({"role": message["role"], "content": content})
        return converted

    converted = []
    for message in messages:
        entry = {"role": message["role"], "content": message.get("content", "")}
        images = message.get("images") or []
        if images:
            # Ollama expects raw base64 (no data: prefix).
            entry["images"] = [
                image.split(",", 1)[1] if image.startswith("data:") else image for image in images
            ]
        converted.append(entry)
    return converted


def chat_request(
    provider: str,
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = True,
    api_key: str = "",
) -> tuple[str, dict, dict]:
    if not model:
        raise ProviderError(
            'No model selected. Click "Load models" in the panel, then pick one. '
            f"Hint: {provider_hint(provider)}"
        )
    if provider == "lmstudio":
        url = endpoint(provider, base_url, "/chat/completions")
        payload = {
            "model": model,
            "messages": to_provider_messages(provider, messages),
            "stream": bool(stream),
            "temperature": float(temperature),
        }
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        return url, _headers(api_key), payload

    url = endpoint(provider, base_url, "/api/chat")
    payload = {
        "model": model,
        "messages": to_provider_messages(provider, messages),
        "stream": bool(stream),
        "options": {"temperature": float(temperature)},
    }
    if max_tokens and max_tokens > 0:
        payload["options"]["num_predict"] = int(max_tokens)
    if provider == "ollama":
        payload["keep_alive"] = "30m"
    return url, _headers(api_key), payload


def parse_chunk(provider: str, raw: str) -> dict | None:
    """Extract one stream line.

    Returns ``{"content": str|None, "reasoning": str|None}`` or ``None`` when the
    line carries no text. Reasoning ("thinking") models such as Qwen or DeepSeek
    stream their chain of thought separately from the answer; both are reported so
    the caller can show the thinking without mixing it into the reply.
    """
    if not raw:
        return None
    if provider == "lmstudio":
        if not raw.startswith("data:"):
            return None
        data = raw[5:].strip()
        if not data or data == "[DONE]":
            return None
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict) and parsed.get("error"):
            raise ProviderError(f"Provider error: {parsed['error']}")
        choices = parsed.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        content = delta.get("content") or None
        reasoning = delta.get("reasoning_content") or delta.get("reasoning") or None
        return _chunk(content, reasoning)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and parsed.get("error"):
        raise ProviderError(f"Provider error: {parsed['error']}")
    message = parsed.get("message") if isinstance(parsed.get("message"), dict) else {}
    content = message.get("content") or parsed.get("response") or None
    reasoning = message.get("thinking") or None
    return _chunk(content, reasoning)


def _chunk(content, reasoning) -> dict | None:
    if not content and not reasoning:
        return None
    return {"content": content or "", "reasoning": reasoning or ""}


def parse_completion(provider: str, data: dict) -> str:
    if not isinstance(data, dict):
        raise ProviderError("Unexpected reply from the provider.")
    if data.get("error"):
        raise ProviderError(f"Provider error: {data['error']}")
    if provider == "lmstudio":
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Empty reply from the provider.")
        return (choices[0].get("message") or {}).get("content") or ""
    return (data.get("message") or {}).get("content") or data.get("response") or ""


def embedding_request(
    provider: str, base_url: str, model: str, texts: list[str], api_key: str = ""
) -> tuple[str, dict, dict]:
    if provider == "lmstudio":
        url = endpoint(provider, base_url, "/embeddings")
        return url, _headers(api_key), {"model": model, "input": texts}
    url = endpoint(provider, base_url, "/api/embed")
    return url, _headers(api_key), {"model": model, "input": texts}


def parse_embeddings(provider: str, data: dict, count: int) -> list[list[float]]:
    if not isinstance(data, dict):
        raise ProviderError("Unexpected embedding response.")
    if data.get("error"):
        raise ProviderError(f"Provider error: {data['error']}")
    if provider == "lmstudio":
        items = data.get("data") or []
        vectors = [item.get("embedding") or [] for item in items]
    else:
        vectors = data.get("embeddings") or []
    if len(vectors) != count:
        raise ProviderError(
            f"Incomplete embeddings: {len(vectors)} vectors received for {count} texts."
        )
    return [list(map(float, vector)) for vector in vectors]


# --------------------------------------------------------------------------- errors


def _raise_http(provider: str, status: int, body: str) -> None:
    detail = (body or "").strip()
    if len(detail) > 400:
        detail = detail[:400] + "..."
    if status in (401, 403):
        hint = (
            "Missing or invalid API key: set it in the ChatHub panel."
            if provider != "lmstudio"
            else "LM Studio refused the request (LM_API_TOKEN?)."
        )
        raise ProviderError(f"HTTP {status} for {provider}. {hint} {detail}")
    if status == 404:
        raise ProviderError(
            f"HTTP 404 for {provider}: unknown endpoint or model. "
            f"Check the model name and the base URL. {detail}"
        )
    raise ProviderError(f"HTTP {status} for {provider}. {detail}")


def _connection_error(provider: str, error: Exception) -> ProviderError:
    reason = connection_reason(error)
    if provider == "ollama_cloud":
        return ProviderError(
            f"Could not reach Ollama Cloud ({reason}). Check your internet connection "
            "and your API key (https://ollama.com/settings/keys)."
        )
    return ProviderError(
        f"Could not reach {provider} ({reason}). Is the local server running? "
        f"{provider_hint(provider)}"
    )


# --------------------------------------------------------------------------- synchronous


def list_models(provider: str, base_url: str, api_key: str = "", timeout: float = 15.0) -> list[str]:
    provider = config.normalize_provider(provider)
    if provider == "lmstudio":
        url = endpoint(provider, base_url, "/models")
    else:
        url = endpoint(provider, base_url, "/api/tags")
    try:
        response = requests.get(url, headers=_headers(api_key), timeout=timeout)
    except requests.RequestException as error:
        raise _connection_error(provider, error) from error
    if response.status_code >= 400:
        _raise_http(provider, response.status_code, response.text)
    try:
        data = response.json()
    except ValueError as error:
        raise ProviderError(f"Unreadable reply from {provider}: {error}") from error
    if provider == "lmstudio":
        names = [item.get("id", "") for item in (data.get("data") or [])]
    else:
        entries = data.get("models") or []
        if provider == "ollama":
            # A local Ollama also advertises the cloud models it can proxy: they carry
            # ``remote_host`` (or a ``-cloud`` suffix on older versions) and do not live
            # on this machine, so they belong to the ollama_cloud list instead.
            entries = [
                entry
                for entry in entries
                if not entry.get("remote_host")
                and not (entry.get("name") or entry.get("model") or "").endswith("-cloud")
            ]
        names = [item.get("name") or item.get("model") or "" for item in entries]
    return sorted({name for name in names if name})


def stream_chat(
    provider: str,
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    api_key: str = "",
    timeout: float = 600.0,
    cancel: "threading.Event | None" = None,
    on_reasoning=None,
) -> Iterator[str]:
    """Yield answer deltas; reasoning deltas go to ``on_reasoning`` if provided."""
    provider = config.normalize_provider(provider)
    url, headers, payload = chat_request(
        provider,
        base_url,
        model,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        api_key=api_key,
    )
    try:
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), stream=True, timeout=(10.0, timeout)
        )
    except requests.RequestException as error:
        raise _connection_error(provider, error) from error
    with response:
        if response.status_code >= 400:
            _raise_http(provider, response.status_code, response.text)
        # Read bytes and decode as UTF-8 ourselves: SSE streams usually carry no
        # charset, and requests would then fall back to latin-1 and mangle accents.
        for raw in response.iter_lines(decode_unicode=False):
            if cancel is not None and cancel.is_set():
                return
            if not raw:
                continue
            chunk = parse_chunk(provider, raw.decode("utf-8", "replace").strip())
            if not chunk:
                continue
            if chunk["reasoning"] and on_reasoning:
                on_reasoning(chunk["reasoning"])
            if chunk["content"]:
                yield chunk["content"]


def complete(
    provider: str,
    base_url: str,
    model: str,
    messages: list[dict],
    **options,
) -> str:
    chunks = list(stream_chat(provider, base_url, model, messages, **options))
    return "".join(chunks).strip()


def embed_texts(
    provider: str,
    base_url: str,
    model: str,
    texts: list[str],
    *,
    api_key: str = "",
    timeout: float = 120.0,
) -> list[list[float]]:
    provider = config.normalize_provider(provider)
    if not texts:
        return []
    if not model:
        raise ProviderError("No embedding model configured (RAG).")
    url, headers, payload = embedding_request(provider, base_url, model, texts, api_key)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.RequestException as error:
        raise _connection_error(provider, error) from error
    if response.status_code == 404 and provider != "lmstudio":
        # Older Ollama servers only expose the single-prompt endpoint.
        vectors = []
        for text in texts:
            single_url = endpoint(provider, base_url, "/api/embeddings")
            try:
                single = requests.post(
                    single_url, headers=headers, data=json.dumps({"model": model, "prompt": text}),
                    timeout=timeout,
                )
            except requests.RequestException as error:
                raise _connection_error(provider, error) from error
            if single.status_code >= 400:
                _raise_http(provider, single.status_code, single.text)
            vectors.append([float(value) for value in (single.json().get("embedding") or [])])
        return vectors
    if response.status_code >= 400:
        _raise_http(provider, response.status_code, response.text)
    try:
        data = response.json()
    except ValueError as error:
        raise ProviderError(f"Unreadable embedding response: {error}") from error
    return parse_embeddings(provider, data, len(texts))


# --------------------------------------------------------------------------- errors


def connection_reason(error: Exception) -> str:
    """Turn a verbose requests/aiohttp exception into a short reason."""
    name = type(error).__name__
    if "Timeout" in name:
        return "delai de connexion depasse"
    if "Connection" in name:
        return "connexion refusee"
    return str(error)[:200]
