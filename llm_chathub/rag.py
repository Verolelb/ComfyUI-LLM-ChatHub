"""Document indexing and retrieval for the RAG part of the chat.

Retrieval is hybrid:

* embeddings from the selected provider when an embedding model is configured and
  reachable (cosine similarity), and
* a self contained BM25 over the same chunks as a fallback, so RAG keeps working
  when no embedding model is available at all.
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import unicodedata
from typing import Callable, Iterable

import numpy as np

from . import providers, store

ProgressFn = Callable[[str, int, int, str], None]

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml",
    ".ini", ".toml", ".cfg", ".conf", ".env", ".py", ".js", ".mjs", ".cjs", ".ts", ".tsx",
    ".jsx", ".html", ".htm", ".css", ".scss", ".xml", ".srt", ".vtt", ".ass", ".tex",
    ".sql", ".sh", ".bat", ".ps1", ".c", ".h", ".hpp", ".cpp", ".cs", ".java", ".go",
    ".rs", ".lua", ".php", ".rb", ".kt", ".swift", ".prompt", ".caption",
}

SKIP_DIRS = {
    ".git", ".svn", "__pycache__", "node_modules", ".venv", "venv", "env", ".idea",
    ".vscode", "dist", "build", ".mypy_cache", ".pytest_cache",
}

_WORD_RE = re.compile(r"[a-z0-9_]{2,}")
_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.DOTALL | re.IGNORECASE)
_SPACE_RE = re.compile(r"[ \t\u00a0]+")
_BLANK_RE = re.compile(r"\n{3,}")

STOPWORDS = {
    "the", "and", "for", "you", "with", "that", "this", "are", "was", "were", "from", "have",
    "has", "had", "not", "but", "les", "des", "une", "que", "qui", "pour", "avec", "dans",
    "sur", "pas", "est", "sont", "plus", "tout", "tous", "cette", "ces", "aux", "par", "comme",
    "elle", "ils", "nous", "vous", "leur", "mais", "donc", "car", "son", "ses", "mon", "mes",
}


# --------------------------------------------------------------------------- text handling


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def tokenize(text: str) -> list[str]:
    tokens = []
    for match in _WORD_RE.findall(fold(text).lower()):
        if match not in STOPWORDS:
            tokens.append(match)
    return tokens


def term_counts(tokens: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def clean_text(raw: str, path: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    extension = os.path.splitext(path)[1].lower()
    if extension in (".html", ".htm", ".xml"):
        text = html.unescape(_TAG_RE.sub(" ", text))
    text = _SPACE_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def read_document(path: str) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return clean_text(handle.read(), path)
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Paragraph aware splitter that never exceeds ``size`` characters per chunk."""
    size = max(200, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if not blocks:
        return []
    chunks: list[str] = []
    current = ""
    for block in blocks:
        while len(block) > size:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(block[:size].strip())
            block = block[max(1, size - overlap):]
        if not current:
            current = block
        elif len(current) + len(block) + 2 <= size:
            current = f"{current}\n\n{block}"
        else:
            chunks.append(current.strip())
            tail = current[-overlap:] if overlap else ""
            # Keep the overlap inside the budget so no chunk exceeds `size`.
            room = max(0, size - len(block) - 2)
            if len(tail) > room:
                tail = tail[-room:] if room else ""
            current = f"{tail}\n\n{block}".strip() if tail else block
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if len(chunk) >= 20]


def iter_files(folder: str, max_file_mb: float = 8.0) -> list[str]:
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Dossier introuvable: {folder}")
    max_bytes = int(max_file_mb * 1024 * 1024)
    found: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".")]
        for name in files:
            if os.path.splitext(name)[1].lower() not in TEXT_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            try:
                if os.path.getsize(path) > max_bytes:
                    continue
            except OSError:
                continue
            found.append(path)
    return sorted(found)


# --------------------------------------------------------------------------- indexing


def index_folder(
    folder: str,
    *,
    provider: str = "lmstudio",
    base_url: str = "",
    api_key: str = "",
    embed_model: str = "",
    chunk_size: int = 1200,
    overlap: int = 200,
    max_file_mb: float = 8.0,
    replace: bool = True,
    progress: ProgressFn | None = None,
) -> dict:
    """Index a folder of text files. Returns a summary dict."""

    def report(stage: str, done: int, total: int, message: str) -> None:
        if progress:
            try:
                progress(stage, done, total, message)
            except Exception:
                pass

    folder = os.path.abspath(os.path.expanduser(folder))
    files = iter_files(folder, max_file_mb)
    if replace:
        store.clear_index()
    report("scan", 0, len(files), f"{len(files)} fichier(s) texte trouve(s)")

    embedding_ok = bool(embed_model)
    embedding_warning = ""
    total_chunks = 0
    indexed_docs = 0
    skipped: list[str] = []

    for position, path in enumerate(files, start=1):
        text = read_document(path)
        if not text or len(text) < 20:
            skipped.append(os.path.basename(path))
            report("index", position, len(files), f"ignore: {os.path.basename(path)}")
            continue

        raw_chunks = chunk_text(text, chunk_size, overlap)
        records: list[dict] = []
        for piece in raw_chunks:
            records.append({"text": piece, "tokens": term_counts(tokenize(piece)), "embedding": None, "dim": 0})

        embedded = False
        if embedding_ok and records:
            for start in range(0, len(records), 32):
                batch = records[start:start + 32]
                try:
                    vectors = providers.embed_texts(
                        provider,
                        base_url,
                        embed_model,
                        [item["text"] for item in batch],
                        api_key=api_key,
                    )
                except providers.ProviderError as error:
                    embedding_ok = False
                    embedding_warning = str(error)
                    break
                for item, vector in zip(batch, vectors):
                    item["embedding"] = np.asarray(vector, dtype=np.float32).tobytes()
                    item["dim"] = len(vector)
            embedded = embedding_ok and all(item["embedding"] is not None for item in records)

        document = store.upsert_document(
            path=path,
            folder=folder,
            size=os.path.getsize(path) if os.path.isfile(path) else 0,
            mtime=os.path.getmtime(path) if os.path.isfile(path) else 0,
            provider=provider if embedded else "",
            model=embed_model if embedded else "",
            chunks=len(records),
            embedded=embedded,
        )
        store.add_chunks(document["id"], records)
        indexed_docs += 1
        total_chunks += len(records)
        report("index", position, len(files), f"indexe: {os.path.basename(path)}")

    stats = store.index_stats()
    report("done", len(files), len(files), f"{total_chunks} extrait(s) prets")
    return {
        "folder": folder,
        "documents": indexed_docs,
        "chunks": total_chunks,
        "embedded": embedding_ok and bool(embed_model),
        "embedding_warning": embedding_warning,
        "skipped": skipped,
        "stats": stats,
    }


# --------------------------------------------------------------------------- retrieval


def _bm25(query_tokens: list[str], rows: list[dict]) -> list[float]:
    if not query_tokens or not rows:
        return [0.0] * len(rows)
    k1, b = 1.5, 0.75
    lengths = []
    for row in rows:
        counts = json.loads(row.get("tokens") or "{}")
        row["_counts"] = counts
        lengths.append(sum(counts.values()))
    average = (sum(lengths) / len(lengths)) or 1.0
    document_frequency: dict[str, int] = {}
    for term in set(query_tokens):
        document_frequency[term] = sum(1 for row in rows if term in row["_counts"])
    scores = []
    for row, length in zip(rows, lengths):
        counts = row["_counts"]
        score = 0.0
        for term in query_tokens:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            df = document_frequency.get(term, 0)
            idf = math.log(1 + (len(rows) - df + 0.5) / (df + 0.5))
            score += idf * (frequency * (k1 + 1)) / (frequency + k1 * (1 - b + b * length / average))
        scores.append(score)
    return scores


def _cosine(query_vector: np.ndarray, rows: list[dict]) -> list[float]:
    scores = []
    query_norm = float(np.linalg.norm(query_vector)) or 1.0
    for row in rows:
        blob = row.get("embedding")
        if not blob:
            scores.append(0.0)
            continue
        vector = np.frombuffer(blob, dtype=np.float32)
        if vector.shape[0] != query_vector.shape[0]:
            scores.append(0.0)
            continue
        denominator = (float(np.linalg.norm(vector)) or 1.0) * query_norm
        scores.append(float(np.dot(vector, query_vector)) / denominator)
    return scores


def retrieve(
    query: str,
    *,
    top_k: int = 4,
    provider: str = "lmstudio",
    base_url: str = "",
    api_key: str = "",
    embed_model: str = "",
    min_score: float = 0.05,
) -> list[dict]:
    """Return the most relevant chunks for ``query``."""
    query = (query or "").strip()
    if not query:
        return []
    rows = list(store.iter_chunk_rows(with_tokens=True))
    if not rows:
        return []

    paths = store.document_paths()
    query_tokens = tokenize(query)
    keyword_scores = _bm25(query_tokens, rows)

    vector_scores = None
    if embed_model:
        try:
            vector = np.asarray(
                providers.embed_texts(
                    provider, base_url, embed_model, [query], api_key=api_key
                )[0],
                dtype=np.float32,
            )
            vector_scores = _cosine(vector, rows)
        except providers.ProviderError:
            vector_scores = None

    def normalize(values: list[float]) -> list[float]:
        top = max(values) if values else 0.0
        if top <= 0:
            return [0.0] * len(values)
        return [value / top for value in values]

    if vector_scores is not None and any(vector_scores):
        keyword_normalized = normalize(keyword_scores)
        combined = [
            0.75 * score + 0.25 * keyword_normalized[index]
            for index, score in enumerate(vector_scores)
        ]
        mode = "hybride"
    else:
        combined = keyword_scores
        mode = "mots-cles"

    ranked = sorted(range(len(rows)), key=lambda index: combined[index], reverse=True)
    results = []
    for index in ranked[: max(1, int(top_k))]:
        score = combined[index]
        if score <= min_score:
            continue
        row = rows[index]
        results.append(
            {
                "text": row["text"],
                "score": round(float(score), 4),
                "source": paths.get(row["document_id"], row["document_id"]),
                "mode": mode,
            }
        )
    return results


def build_context(results: list[dict], max_chars: int = 6000) -> str:
    """Format retrieved chunks as a block ready to inject in the system prompt."""
    if not results:
        return ""
    blocks = []
    used = 0
    for result in results:
        source = os.path.basename(result.get("source") or "document")
        block = f"[{source}]\n{result['text']}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block) + 2
    if not blocks:
        return ""
    return "\n\n".join(blocks)
