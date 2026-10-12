"""Repo Map: a compressed, queryable view of a repository.

Built once per repository state and cached on disk. Each file gets a language,
its exported symbols and a one-line description; the rendered map is squeezed
into a token budget so it can sit in an agent's context. Two retrieval
strategies find relevant files for a task:

- keyword: scores the map entries (paths, symbols, descriptions); fast, no index
- embedding: semantic search over file-content chunks in ChromaDB; slower, catches paraphrases

Changed files are detected with `git diff` against the commit the cache was
built from, so only those are re-summarized and re-indexed.
"""

import ast
import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

DEFAULT_TOKEN_BUDGET = 4000
DEFAULT_TOP_K = 10
CHUNK_LINES = 40

LANGUAGES = {
    ".py": "python", ".md": "markdown", ".txt": "text", ".toml": "toml", ".ini": "ini",
    ".cfg": "ini", ".yml": "yaml", ".yaml": "yaml", ".json": "json", ".js": "javascript",
    ".ts": "typescript", ".sh": "shell",
}
_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


@dataclass
class FileEntry:
    path: str
    language: str
    symbols: list[str] = field(default_factory=list)
    description: str = ""
    digest: str = ""


@dataclass
class RepoMap:
    root: Path
    head: str
    files: dict[str, FileEntry]

    def render(self, token_budget: int = DEFAULT_TOKEN_BUDGET) -> str:
        """Render the map, dropping detail until it fits the budget."""
        for level in ("full", "short_symbols", "no_test_detail", "paths_and_descriptions", "paths_only"):
            text = self._render(level)
            if count_tokens(text) <= token_budget:
                return text
        lines, total = [], 0
        for line in text.splitlines():  # last resort: hard cut, but say so
            total += count_tokens(line) + 1
            if total > token_budget - 20:
                lines.append(f"... ({len(self.files)} files total, map truncated to fit budget)")
                break
            lines.append(line)
        return "\n".join(lines)

    def _render(self, level: str) -> str:
        out = [f"# Repo map ({len(self.files)} files)"]
        for path in sorted(self.files):
            e = self.files[path]
            is_test = path.startswith("tests/") or Path(path).name.startswith("test_")
            line = f"{path} [{e.language}]"
            if level != "paths_only" and e.description and not (level == "no_test_detail" and is_test):
                line += f" - {e.description}"
            symbols = e.symbols
            if level == "short_symbols" or (level == "no_test_detail" and not is_test):
                symbols = symbols[:6]
            if level in ("full", "short_symbols") or (level == "no_test_detail" and not is_test):
                if symbols:
                    more = len(e.symbols) - len(symbols)
                    line += " :: " + ", ".join(symbols) + (f" (+{more})" if more > 0 else "")
            out.append(line)
        return "\n".join(out)

    def keyword_search(self, query: str, k: int = DEFAULT_TOP_K) -> list[str]:
        terms = _terms(query)
        scored = []
        for path, e in self.files.items():
            fields = (
                (3.0, _terms(path)),
                (2.0, {t for s in e.symbols for t in _terms(s)}),
                (1.0, _terms(e.description)),
            )
            score = sum(weight * len(terms & words) for weight, words in fields)
            if score:
                scored.append((score, path))
        return [p for _, p in sorted(scored, key=lambda x: (-x[0], x[1]))[:k]]

    def to_json(self) -> str:
        return json.dumps({"head": self.head, "files": {p: asdict(e) for p, e in self.files.items()}}, indent=1)


def _terms(text: str) -> set[str]:
    """Lowercase word set, splitting snake_case, camelCase and paths."""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9]+", text.lower())
    terms = set(words)
    terms |= {w[:-1] for w in words if w.endswith("s") and len(w) > 3}  # crude plural folding
    return terms - {"the", "and", "for", "with", "from", "this", "that", "should", "when", "app", "py"}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def list_files(root: Path) -> list[str]:
    tracked = _git(root, "ls-files").splitlines()
    untracked = _git(root, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted(p for p in set(tracked + untracked) if (root / p).is_file())


def changed_since(root: Path, head: str) -> set[str]:
    """Files changed in the working tree or commits since `head`, plus untracked files."""
    changed = set(_git(root, "diff", "--name-only", head).splitlines())
    changed |= set(_git(root, "ls-files", "--others", "--exclude-standard").splitlines())
    return changed


def describe_python(source: str) -> tuple[list[str], str]:
    """Top-level public symbols (classes with their public methods) and the module docstring's first line."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], ""
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            symbols.append(f"{node.name}()")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [n.name for n in node.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
            symbols.append(node.name + (f"[{', '.join(methods)}]" if methods else ""))
        elif isinstance(node, ast.Assign):
            symbols += [t.id for t in node.targets if isinstance(t, ast.Name) and t.id.isupper()]
    doc = (ast.get_docstring(tree) or "").strip().splitlines()
    return symbols, (doc[0] if doc else "")


def build_entry(root: Path, path: str, summarize: Callable[[str, str], str] | None) -> FileEntry:
    raw = (root / path).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    entry = FileEntry(path=path, language=LANGUAGES.get(Path(path).suffix, "other"),
                      digest=hashlib.sha1(raw).hexdigest())
    if entry.language == "python":
        entry.symbols, entry.description = describe_python(text)
    elif entry.language == "markdown":
        heading = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), "")
        entry.description = heading
    if not entry.description and summarize and text.strip():
        entry.description = summarize(path, text[:4000])
    return entry


def build_repo_map(
    root: Path,
    cache_dir: Path | None = None,
    summarize: Callable[[str, str], str] | None = None,
) -> RepoMap:
    """Build the map, reusing cached entries for files unchanged since the cached commit."""
    root = root.resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    cache_file = cache_dir / f"repo_map_{hashlib.sha1(str(root).encode()).hexdigest()[:10]}.json" if cache_dir else None

    cached: dict[str, FileEntry] = {}
    stale: set[str] | None = None
    if cache_file and cache_file.exists():
        data = json.loads(cache_file.read_text())
        cached = {p: FileEntry(**e) for p, e in data["files"].items()}
        try:
            stale = changed_since(root, data["head"])
        except subprocess.CalledProcessError:  # cached commit no longer exists: rebuild everything
            cached, stale = {}, None

    files = {}
    for path in list_files(root):
        reuse = path in cached and stale is not None and path not in stale
        files[path] = cached[path] if reuse else build_entry(root, path, summarize)

    repo_map = RepoMap(root=root, head=head, files=files)
    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(repo_map.to_json())
    return repo_map


class EmbeddingIndex:
    """Semantic file search over content chunks in a persistent ChromaDB collection.

    Uses Chroma's built-in ONNX MiniLM embedder, so no PyTorch install is needed.
    Chunks are keyed by file digest: unchanged files are never re-embedded.
    """

    def __init__(self, repo_map: RepoMap, persist_dir: Path) -> None:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
        name = "repo_" + hashlib.sha1(str(repo_map.root).encode()).hexdigest()[:12]
        self.collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self.repo_map = repo_map
        self._sync()

    def _sync(self) -> None:
        existing = self.collection.get(include=["metadatas"])
        indexed = {m["path"]: m["digest"] for m in existing["metadatas"]}
        stale = [p for p, d in indexed.items() if self.repo_map.files.get(p, FileEntry("", "")).digest != d]
        if stale:
            self.collection.delete(where={"path": {"$in": stale}})
        ids, docs, metas = [], [], []
        for path, entry in self.repo_map.files.items():
            if indexed.get(path) == entry.digest and path not in stale:
                continue
            lines = (self.repo_map.root / path).read_text(errors="replace").splitlines()
            for start in range(0, max(len(lines), 1), CHUNK_LINES):
                chunk = "\n".join(lines[start:start + CHUNK_LINES])
                if chunk.strip():
                    ids.append(f"{path}:{start}")
                    docs.append(f"{path}\n{chunk}")
                    metas.append({"path": path, "digest": entry.digest, "start_line": start + 1})
        if ids:
            self.collection.add(ids=ids, documents=docs, metadatas=metas)

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[str]:
        n = min(self.collection.count(), k * 4)
        if not n:
            return []
        hits = self.collection.query(query_texts=[query], n_results=n, include=["metadatas", "distances"])
        best: dict[str, float] = {}
        for meta, dist in zip(hits["metadatas"][0], hits["distances"][0]):
            best[meta["path"]] = min(dist, best.get(meta["path"], 2.0))
        return sorted(best, key=best.get)[:k]
