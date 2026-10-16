#!/usr/bin/env python3
"""
Document Preprocessor for a Digital Twin Knowledge Base (RAG-ready)
===================================================================

Pipeline overview (each step is a clearly marked section below):

    1. INGEST      -> read .txt / .md / .log files, detect their type
    2. PARSE       -> turn raw text into structured blocks
                      (headings, paragraphs, lists, tables, code blocks, log entries)
    3. EXTRACT     -> pull out structured data (front matter, outline, tables,
                      code blocks, links, entities, log statistics)
    4. CLEAN       -> normalize unicode, whitespace, quotes, control characters
    5. CHUNK       -> structure-aware chunking sized for RAG, with overlap
    6. METADATA    -> attach rich metadata to every chunk (keywords, entities, ...)
    7. INDEX       -> build a BM25 inverted index + metadata filter indices
    8. QUALITY     -> compute quality metrics and an overall score
    9. SEARCH      -> query the index (with optional metadata filters)

Usage examples:
    python document_preprocessor.py --demo
    python document_preprocessor.py --input ./docs --output ./kb_output
    python document_preprocessor.py --output ./kb_output --search "database timeout"
    python document_preprocessor.py --output ./kb_output --search "install" --has-code

Only the Python standard library is required (Python 3.8+).
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================
@dataclass
class Config:
    max_tokens: int = 200          # target upper bound for a prose chunk
    min_tokens: int = 30           # chunks smaller than this get merged if possible
    overlap_sentences: int = 1     # sentences repeated between split chunks
    atomic_factor: float = 1.5     # code/tables may exceed max_tokens by this factor
                                   # so we avoid cutting them in half
    keywords_per_chunk: int = 6    # how many keywords to attach to each chunk


SUPPORTED_EXTENSIONS = {".txt": "text", ".md": "markdown", ".markdown": "markdown",
                        ".log": "log", ".rst": "text"}

# Very small stopword list - enough to keep keywords meaningful without a dependency
STOPWORDS = set("""a an and are as at be but by for from has have he her his i if in into is it
its of on or our she so that the their them then there these they this to was we were what when
which who will with you your not no can do does did than also about more most other some such
only own same too very just over under again further once here how all any both each few""".split())


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class Document:
    """A raw document after ingestion."""
    doc_id: str
    path: str
    doc_type: str          # markdown | text | log
    raw_text: str
    modified: str


@dataclass
class Block:
    """One structural unit of a document (a paragraph, a table, a code block, ...)."""
    kind: str              # heading | paragraph | list | table | code | log
    text: str
    section: str = ""      # heading path such as "Setup > Installation"
    extra: dict = field(default_factory=dict)


@dataclass
class Piece:
    """Intermediate chunk, before metadata is attached."""
    text: str
    section: str
    kinds: set
    languages: list
    log_meta: list


@dataclass
class Chunk:
    """Final RAG chunk with metadata."""
    chunk_id: str
    doc_id: str
    source: str
    doc_type: str
    doc_title: str
    chunk_index: int
    section: str
    text: str                       # the clean chunk text
    contextual_text: str            # text prefixed with title/section (good for embeddings)
    token_count: int
    char_count: int
    content_hash: str
    kinds: List[str]
    has_code: bool
    has_table: bool
    code_languages: List[str]
    keywords: List[str] = field(default_factory=list)
    entities: Dict[str, List[str]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    log_levels: Dict[str, int] = field(default_factory=dict)
    prev_id: Optional[str] = None
    next_id: Optional[str] = None
    created_at: str = ""


# =============================================================================
# SMALL SHARED HELPERS
# =============================================================================
def count_tokens(text: str) -> int:
    """Cheap tokenizer approximation: words and punctuation marks each count as 1.
    (Swap this for tiktoken or your embedding model's tokenizer in production.)"""
    return len(re.findall(r"\w+|[^\w\s]", text))


def words(text: str) -> List[str]:
    """Lowercased word tokens used for search and keyword extraction."""
    return re.findall(r"[a-z0-9_]{2,}", text.lower())


# Regexes shared by several steps
TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
LOG_RE = re.compile(
    r"^\[?(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\]?"
    r"\s+\[?(?P<level>TRACE|DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]?:?\s*(?P<msg>.*)$",
    re.IGNORECASE)


# =============================================================================
# STEP 1: INGEST
# =============================================================================
def read_file(path: Path) -> str:
    """Read a file trying several encodings so odd files don't crash the pipeline."""
    data = path.read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def normalize_raw(text: str) -> str:
    """Unify line endings and drop null bytes before any parsing happens."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")


def detect_type(path: Path, text: str) -> str:
    """Decide the document type from the extension, sniffing .txt files for log format."""
    doc_type = SUPPORTED_EXTENSIONS.get(path.suffix.lower(), "text")
    if doc_type == "text":
        sample = [l for l in text.split("\n") if l.strip()][:50]
        if sample and sum(bool(LOG_RE.match(l)) for l in sample) / len(sample) >= 0.3:
            return "log"   # looks like a log even though it has a .txt extension
    return doc_type


def ingest(input_dir: Path) -> List[Document]:
    """Walk the input folder and load every supported file."""
    docs = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = normalize_raw(read_file(path))
        rel = str(path.relative_to(input_dir))
        docs.append(Document(
            doc_id=hashlib.sha1(rel.encode()).hexdigest()[:8],   # stable ID from the path
            path=rel,
            doc_type=detect_type(path, text),
            raw_text=text,
            modified=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        ))
    return docs


# =============================================================================
# STEP 2: PARSE (raw text -> structured blocks)
# =============================================================================
def split_table_row(row: str) -> List[str]:
    """'| a | b |' -> ['a', 'b']"""
    return [c.strip() for c in row.strip().strip("|").split("|")]


def parse_table(rows: List[str]) -> dict:
    """Convert markdown table lines into {'header': [...], 'rows': [{col: value}, ...]}."""
    header = split_table_row(rows[0])
    data = [dict(zip(header, split_table_row(r))) for r in rows[2:]]
    return {"header": header, "rows": data}


def parse_markdown(text: str) -> Tuple[List[Block], dict]:
    """Line-by-line state machine that recognises front matter, headings, fenced code,
    tables, lists and paragraphs. Returns (blocks, front_matter)."""
    lines = text.split("\n")
    n, i = len(lines), 0
    front: dict = {}

    # --- YAML-style front matter between '---' lines at the very top ---
    if lines and lines[0].strip() == "---":
        for j in range(1, n):
            if lines[j].strip() == "---":
                for l in lines[1:j]:
                    if ":" in l:
                        k, v = l.split(":", 1)
                        front[k.strip()] = v.strip().strip("'\"")
                i = j + 1
                break

    blocks: List[Block] = []
    path: List[Tuple[int, str]] = []   # heading stack: (level, title)
    para: List[str] = []               # lines of the paragraph being collected

    def section() -> str:
        return " > ".join(t for _, t in path)

    def flush_para():
        """Close the current paragraph, joining soft-wrapped lines into one line."""
        if para:
            blocks.append(Block("paragraph", " ".join(l.strip() for l in para), section()))
            para.clear()

    while i < n:
        line = lines[i]

        # --- fenced code block (``` or ~~~) kept verbatim ---
        fence = re.match(r"^\s*(```|~~~)\s*([\w+#.\-]*)", line)
        if fence:
            flush_para()
            marker, lang = fence.group(1), fence.group(2) or "text"
            body = []
            i += 1
            while i < n and not lines[i].strip().startswith(marker):
                body.append(lines[i])
                i += 1
            i += 1  # skip the closing fence
            if any(b.strip() for b in body):
                blocks.append(Block("code", "\n".join(body), section(), {"language": lang}))
            continue

        # --- heading: updates the section path used as context for later blocks ---
        h = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if h:
            flush_para()
            level, title = len(h.group(1)), h.group(2)
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
            blocks.append(Block("heading", title, section(), {"level": level}))
            i += 1
            continue

        # --- table: a '|' row followed by a separator row like |---|---| ---
        if "|" in line and i + 1 < n and "|" in lines[i + 1] and TABLE_SEP_RE.match(lines[i + 1]):
            flush_para()
            rows = []
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(lines[i])
                i += 1
            blocks.append(Block("table", "\n".join(rows), section(), {"table": parse_table(rows)}))
            continue

        # --- list: consecutive bullet/numbered lines (plus indented continuations) ---
        if LIST_ITEM_RE.match(line):
            flush_para()
            items = []
            while i < n and lines[i].strip() and (LIST_ITEM_RE.match(lines[i]) or lines[i][0] in " \t"):
                items.append(lines[i].rstrip())
                i += 1
            blocks.append(Block("list", "\n".join(items), section()))
            continue

        # --- blank line ends a paragraph; anything else extends it ---
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1

    flush_para()
    return blocks, front


def parse_text(text: str) -> List[Block]:
    """Plain text: paragraphs separated by blank lines. Short ALL-CAPS or 'Title:' lines
    are treated as headings so we still get a usable section path."""
    blocks, path = [], ""
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        one_line = "\n" not in para
        if one_line and len(para) <= 60 and (para.isupper() or para.endswith(":")):
            path = para.rstrip(":").strip()
            blocks.append(Block("heading", path, path, {"level": 1}))
        else:
            blocks.append(Block("paragraph", " ".join(l.strip() for l in para.split("\n")), path))
    return blocks


def normalize_level(level: str) -> str:
    level = level.upper()
    return {"WARNING": "WARN", "FATAL": "CRITICAL"}.get(level, level)


def parse_log(text: str) -> List[Block]:
    """Parse log lines into entries. Lines that don't start with a timestamp
    (stack traces, wrapped messages) are attached to the previous entry.
    Consecutive identical messages are collapsed into one entry with a repeat count."""
    entries: List[dict] = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        m = LOG_RE.match(line)
        if m:
            entries.append({"ts": m.group("ts").replace(",", "."),
                            "level": normalize_level(m.group("level")),
                            "msg": m.group("msg").strip(), "count": 1})
        elif entries:
            entries[-1]["msg"] += "\n" + line.rstrip()          # continuation line
        else:
            entries.append({"ts": None, "level": "UNKNOWN", "msg": line.strip(), "count": 1})

    # Collapse consecutive duplicates (noise reduction for RAG)
    collapsed: List[dict] = []
    for e in entries:
        if collapsed and collapsed[-1]["msg"] == e["msg"] and collapsed[-1]["level"] == e["level"]:
            collapsed[-1]["count"] += 1
        else:
            collapsed.append(e)

    blocks = []
    for e in collapsed:
        suffix = f" (repeated x{e['count']})" if e["count"] > 1 else ""
        text = f"{e['ts'] or ''} {e['level']} {e['msg']}{suffix}".strip()
        blocks.append(Block("log", text, "", {"ts": e["ts"], "level": e["level"], "count": e["count"]}))
    return blocks


# =============================================================================
# STEP 4: CLEAN (run per block so code blocks are never damaged)
# =============================================================================
SMART_CHARS = {"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
               "\u2013": "-", "\u2014": "-", "\u00a0": " "}
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def clean_text(text: str, collapse_ws: bool = True) -> str:
    """Normalize unicode, quotes, invisible characters and whitespace."""
    text = unicodedata.normalize("NFKC", text)            # unify look-alike characters
    for bad, good in SMART_CHARS.items():                 # curly quotes / dashes -> ASCII
        text = text.replace(bad, good)
    text = ZERO_WIDTH_RE.sub("", text)                    # invisible characters
    text = CONTROL_RE.sub("", text)                       # control characters
    text = HTML_COMMENT_RE.sub("", text)                  # <!-- comments -->
    if collapse_ws:
        text = re.sub(r"[ \t]+", " ", text)               # multiple spaces -> one
    return "\n".join(l.rstrip() for l in text.split("\n")).strip()


def clean_blocks(blocks: List[Block]) -> List[Block]:
    """Clean each block according to its kind."""
    cleaned = []
    for b in blocks:
        if b.kind == "code":
            # Code must stay verbatim: only strip trailing whitespace on each line
            b.text = "\n".join(l.rstrip() for l in b.text.split("\n")).strip("\n")
        elif b.kind == "log":
            b.text = clean_text(b.text, collapse_ws=False)   # keep stack-trace indentation
        elif b.kind == "table":
            b.text = clean_text(b.text)
            b.extra["table"] = parse_table(b.text.split("\n"))   # re-parse the cleaned table
        else:
            b.text = clean_text(b.text)
        if b.text:                                           # drop blocks that became empty
            cleaned.append(b)
    return cleaned


# =============================================================================
# STEP 3: EXTRACT structured data (entities + per-document summary)
# =============================================================================
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
URL_RE = re.compile(r"https?://[^\s)>\]\"']+")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def extract_entities(text: str) -> Dict[str, List[str]]:
    """Find emails, URLs, IPs and ISO dates. Empty categories are omitted."""
    found = {"emails": EMAIL_RE.findall(text), "urls": URL_RE.findall(text),
             "ips": IP_RE.findall(text), "dates": DATE_RE.findall(text)}
    return {k: sorted(set(v)) for k, v in found.items() if v}


def extract_document_data(doc: Document, blocks: List[Block], front: dict) -> dict:
    """Collect everything structured we learned about one document."""
    info = {
        "doc_id": doc.doc_id, "source": doc.path, "doc_type": doc.doc_type,
        "modified": doc.modified, "front_matter": front,
        "outline": [{"level": b.extra["level"], "title": b.text} for b in blocks if b.kind == "heading"],
        "tables": [b.extra["table"] for b in blocks if b.kind == "table"],
        "code_blocks": [{"language": b.extra["language"], "lines": b.text.count("\n") + 1}
                        for b in blocks if b.kind == "code"],
        "links": re.findall(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", doc.raw_text),
        "entities": extract_entities("\n".join(b.text for b in blocks)),
        "raw_chars": len(doc.raw_text),
    }
    if doc.doc_type == "log":
        logs = [b for b in blocks if b.kind == "log"]
        stamps = [b.extra["ts"] for b in logs if b.extra["ts"]]
        info["log_summary"] = {
            "entries": len(logs),
            "levels": dict(Counter(b.extra["level"] for b in logs)),
            "time_start": min(stamps) if stamps else None,
            "time_end": max(stamps) if stamps else None,
        }
    # Pick a title: front matter > first heading > file name
    first_h = next((b.text for b in blocks if b.kind == "heading"), None)
    info["title"] = front.get("title") or first_h or Path(doc.path).stem
    return info


# =============================================================================
# STEP 5: CHUNK (structure-aware)
# =============================================================================
def split_long_text(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Split an oversized paragraph/list into sentence-based pieces with overlap."""
    sep = "\n" if "\n" in text else " "
    units: List[str] = []
    for line in text.split("\n"):
        units.extend(re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", line))

    # A "sentence" with no punctuation that is still too long gets cut by words
    safe: List[str] = []
    for u in units:
        if count_tokens(u) > max_tokens:
            w = u.split()
            safe.extend(" ".join(w[k:k + max_tokens]) for k in range(0, len(w), max_tokens))
        else:
            safe.append(u)

    pieces, cur, tok = [], [], 0
    for s in safe:
        t = count_tokens(s)
        if cur and tok + t > max_tokens:
            pieces.append(sep.join(cur))
            cur = cur[-overlap:] if overlap else []           # carry overlap into next chunk
            tok = sum(count_tokens(x) for x in cur)
            if tok + t > max_tokens:                          # overlap too big -> drop it
                cur, tok = [], 0
        cur.append(s)
        tok += t
    if cur:
        pieces.append(sep.join(cur))
    return pieces


def split_code(code: str, lang: str, limit: int) -> List[str]:
    """Return fenced code strings. Big code blocks are split by lines; every part
    gets its own balanced fence so formatting survives."""
    if count_tokens(code) <= limit:
        return [f"```{lang}\n{code}\n```"]
    parts, cur, tok = [], [], 0
    for line in code.split("\n"):
        t = count_tokens(line)
        if cur and tok + t > limit:
            parts.append(f"```{lang}\n" + "\n".join(cur) + "\n```")
            cur, tok = [], 0
        cur.append(line)
        tok += t
    if cur:
        parts.append(f"```{lang}\n" + "\n".join(cur) + "\n```")
    return parts


def split_table(table_text: str, limit: int) -> List[str]:
    """Large tables are split by rows, repeating the header + separator in every part."""
    if count_tokens(table_text) <= limit:
        return [table_text]
    lines = table_text.split("\n")
    head, rows = lines[:2], lines[2:]
    parts, cur, tok = [], [], count_tokens("\n".join(head))
    base = tok
    for r in rows:
        t = count_tokens(r)
        if cur and tok + t > limit:
            parts.append("\n".join(head + cur))
            cur, tok = [], base
        cur.append(r)
        tok += t
    if cur:
        parts.append("\n".join(head + cur))
    return parts


def merge_small(pieces: List[Piece], cfg: Config) -> List[Piece]:
    """Merge undersized neighbours that share a section, as long as they stay under max_tokens."""
    out: List[Piece] = []
    for p in pieces:
        if (out and out[-1].section == p.section and "log" not in p.kinds
                and (count_tokens(out[-1].text) < cfg.min_tokens or count_tokens(p.text) < cfg.min_tokens)
                and count_tokens(out[-1].text) + count_tokens(p.text) <= cfg.max_tokens):
            out[-1].text += "\n\n" + p.text
            out[-1].kinds |= p.kinds
            out[-1].languages += p.languages
        else:
            out.append(p)
    return out


def chunk_blocks(blocks: List[Block], cfg: Config) -> List[Piece]:
    """Pack paragraphs, lists, code and tables into chunks.
    - chunks never cross a section (heading) boundary
    - code blocks and tables are treated as atomic units (split only if huge)
    - explanatory text stays in the same chunk as the code/table that follows when it fits"""
    pieces: List[Piece] = []
    st = {"texts": [], "tok": 0, "section": None, "kinds": set(), "langs": []}

    def flush():
        if st["texts"]:
            pieces.append(Piece("\n\n".join(st["texts"]), st["section"] or "",
                                set(st["kinds"]), list(st["langs"]), []))
        st.update(texts=[], tok=0, section=None, kinds=set(), langs=[])

    def add(text: str, kind: str, section: str, lang: Optional[str] = None):
        tok = count_tokens(text)
        limit = cfg.max_tokens if kind in ("paragraph", "list") else cfg.max_tokens * cfg.atomic_factor
        if st["texts"] and (section != st["section"] or st["tok"] + tok > limit):
            flush()
        st["section"] = section
        st["texts"].append(text)
        st["tok"] += tok
        st["kinds"].add(kind)
        if lang:
            st["langs"].append(lang)

    atomic_limit = cfg.max_tokens * cfg.atomic_factor
    for b in blocks:
        if b.kind == "heading":
            flush()                                  # new section = new chunk
        elif b.kind in ("paragraph", "list"):
            if count_tokens(b.text) > cfg.max_tokens:
                for part in split_long_text(b.text, cfg.max_tokens, cfg.overlap_sentences):
                    add(part, b.kind, b.section)
            else:
                add(b.text, b.kind, b.section)
        elif b.kind == "code":
            lang = b.extra.get("language", "text")
            for part in split_code(b.text, lang, atomic_limit):
                add(part, "code", b.section, lang)
        elif b.kind == "table":
            for part in split_table(b.text, atomic_limit):
                add(part, "table", b.section)
    flush()
    return merge_small(pieces, cfg)


def chunk_logs(blocks: List[Block], cfg: Config) -> List[Piece]:
    """Logs are chunked by consecutive entries (keeping chronological order)."""
    pieces, cur, meta, tok = [], [], [], 0

    def flush():
        if cur:
            pieces.append(Piece("\n".join(cur), "", {"log"}, [], list(meta)))
            cur.clear(); meta.clear()

    for b in blocks:
        t = count_tokens(b.text)
        if cur and tok + t > cfg.max_tokens:
            flush()
            tok = 0
        cur.append(b.text)
        meta.append(b.extra)
        tok += t
    flush()
    return pieces


# =============================================================================
# STEP 6: METADATA (turn pieces into full Chunk objects)
# =============================================================================
def build_chunks(doc_info: dict, front: dict, pieces: List[Piece]) -> List[Chunk]:
    """Attach IDs, hashes, context text, entities, tags and log info to each piece."""
    now = datetime.now(timezone.utc).isoformat()
    tags = [t.strip() for t in re.split(r"[,\[\]]", str(front.get("tags", ""))) if t.strip()]
    chunks: List[Chunk] = []

    for idx, p in enumerate(pieces):
        ctx_prefix = f"[{doc_info['title']} | {p.section}]" if p.section else f"[{doc_info['title']}]"
        stamps = [m["ts"] for m in p.log_meta if m.get("ts")]
        levels: Dict[str, int] = {}
        for m in p.log_meta:
            levels[m["level"]] = levels.get(m["level"], 0) + m.get("count", 1)

        chunks.append(Chunk(
            chunk_id=f"{doc_info['doc_id']}-{idx:04d}",
            doc_id=doc_info["doc_id"], source=doc_info["source"], doc_type=doc_info["doc_type"],
            doc_title=doc_info["title"], chunk_index=idx, section=p.section,
            text=p.text, contextual_text=f"{ctx_prefix}\n{p.text}",
            token_count=count_tokens(p.text), char_count=len(p.text),
            content_hash=hashlib.md5(p.text.encode()).hexdigest(),
            kinds=sorted(p.kinds), has_code="code" in p.kinds, has_table="table" in p.kinds,
            code_languages=sorted(set(p.languages)),
            entities=extract_entities(p.text), tags=tags,
            time_start=min(stamps) if stamps else None,
            time_end=max(stamps) if stamps else None,
            log_levels=levels, created_at=now,
        ))

    # Link neighbours so a retriever can fetch surrounding context later
    for i, c in enumerate(chunks):
        c.prev_id = chunks[i - 1].chunk_id if i > 0 else None
        c.next_id = chunks[i + 1].chunk_id if i < len(chunks) - 1 else None
    return chunks


def add_keywords(chunks: List[Chunk], cfg: Config) -> None:
    """TF-IDF keywords across the whole corpus (needs all chunks, so runs after chunking)."""
    n = len(chunks)
    token_lists = [[w for w in words(c.contextual_text) if w not in STOPWORDS and not w.isdigit()]
                   for c in chunks]
    df = Counter(w for toks in token_lists for w in set(toks))
    for c, toks in zip(chunks, token_lists):
        tf = Counter(toks)
        scored = {w: f * math.log(1 + n / df[w]) for w, f in tf.items()}
        c.keywords = [w for w, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:cfg.keywords_per_chunk]]


# =============================================================================
# STEP 7: INDEX (BM25 inverted index + metadata indices)
# =============================================================================
def build_index(chunks: List[Chunk]) -> dict:
    """Create a JSON-serialisable search index."""
    postings: Dict[str, Dict[str, int]] = defaultdict(dict)   # term -> {chunk_id: term_freq}
    lengths: Dict[str, int] = {}
    meta = {"by_doc": defaultdict(list), "by_type": defaultdict(list),
            "by_keyword": defaultdict(list), "by_level": defaultdict(list),
            "by_language": defaultdict(list), "by_tag": defaultdict(list)}

    for c in chunks:
        toks = words(c.contextual_text) + c.keywords           # keywords get a small boost
        lengths[c.chunk_id] = len(toks)
        for term, freq in Counter(toks).items():
            postings[term][c.chunk_id] = freq
        meta["by_doc"][c.doc_id].append(c.chunk_id)
        meta["by_type"][c.doc_type].append(c.chunk_id)
        for k in c.keywords:
            meta["by_keyword"][k].append(c.chunk_id)
        for lvl in c.log_levels:
            meta["by_level"][lvl].append(c.chunk_id)
        for lang in c.code_languages:
            meta["by_language"][lang].append(c.chunk_id)
        for t in c.tags:
            meta["by_tag"][t.lower()].append(c.chunk_id)

    return {"N": len(chunks),
            "avgdl": (sum(lengths.values()) / len(lengths)) if lengths else 0.0,
            "lengths": lengths, "postings": postings,
            "meta": {k: dict(v) for k, v in meta.items()}}


# =============================================================================
# STEP 9: SEARCH (BM25 ranking with optional metadata filters)
# =============================================================================
def search(index: dict, chunks_by_id: Dict[str, dict], query: str, top_k: int = 5,
           doc_type: Optional[str] = None, has_code: bool = False,
           level: Optional[str] = None) -> List[Tuple[float, dict]]:
    """Rank chunks for a query using BM25 (k1=1.5, b=0.75) after applying filters."""
    k1, b = 1.5, 0.75
    N, avgdl = index["N"], index["avgdl"] or 1.0
    scores: Dict[str, float] = defaultdict(float)

    for term in set(words(query)):
        posting = index["postings"].get(term)
        if not posting:
            continue
        idf = math.log(1 + (N - len(posting) + 0.5) / (len(posting) + 0.5))
        for cid, tf in posting.items():
            dl = index["lengths"][cid]
            scores[cid] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    results = []
    for cid, score in scores.items():
        c = chunks_by_id[cid]
        if doc_type and c["doc_type"] != doc_type:
            continue
        if has_code and not c["has_code"]:
            continue
        if level and level.upper() not in c["log_levels"]:
            continue
        results.append((score, c))
    return sorted(results, key=lambda x: -x[0])[:top_k]


# =============================================================================
# STEP 8: QUALITY METRICS
# =============================================================================
def compute_quality(chunks: List[Chunk], doc_blocks: Dict[str, List[Block]],
                    doc_infos: List[dict], cfg: Config) -> dict:
    """Measure how good the output is and combine the metrics into a 0-100 score."""
    if not chunks:
        return {"error": "no chunks produced", "overall_score": 0}

    tokens = [c.token_count for c in chunks]
    hi = cfg.max_tokens * cfg.atomic_factor

    # 1) Size distribution: how many chunks fall in the desired range?
    in_range = sum(cfg.min_tokens <= t <= hi for t in tokens) / len(tokens)

    # 2) Uniqueness: duplicate chunks waste index space and bias retrieval
    uniqueness = len({c.content_hash for c in chunks}) / len(chunks)

    # 3) Coverage: does every word of the cleaned content appear in some chunk?
    coverage_scores = []
    for doc_id, blocks in doc_blocks.items():
        src = {w for b in blocks if b.kind != "heading" for w in words(b.text)}
        got = {w for c in chunks if c.doc_id == doc_id for w in words(c.text)}
        coverage_scores.append(len(src & got) / len(src) if src else 1.0)
    coverage = sum(coverage_scores) / len(coverage_scores)

    # 4) Structure integrity: code fences balanced and tables keep their header separator
    code_chunks = [c for c in chunks if c.has_code]
    table_chunks = [c for c in chunks if c.has_table]
    fence_ok = sum(c.text.count("```") % 2 == 0 for c in code_chunks)
    table_ok = sum(any(TABLE_SEP_RE.match(l) and "|" in l for l in c.text.split("\n"))
                   for c in table_chunks)
    total_struct = len(code_chunks) + len(table_chunks)
    structure = (fence_ok + table_ok) / total_struct if total_struct else 1.0

    # 5) Metadata completeness
    required = ["chunk_id", "doc_id", "source", "doc_type", "doc_title",
                "token_count", "content_hash", "keywords"]
    filled = sum(bool(getattr(c, f)) for c in chunks for f in required)
    completeness = filled / (len(chunks) * len(required))

    # Weighted overall score
    score = 100 * (0.30 * in_range + 0.20 * coverage + 0.15 * structure +
                   0.15 * uniqueness + 0.20 * completeness)

    return {
        "documents": len(doc_infos),
        "chunks": len(chunks),
        "chunks_by_type": dict(Counter(c.doc_type for c in chunks)),
        "tokens": {"min": min(tokens), "max": max(tokens),
                   "mean": round(statistics.mean(tokens), 1),
                   "median": statistics.median(tokens)},
        "undersized_chunks": sum(t < cfg.min_tokens for t in tokens),
        "oversized_chunks": sum(t > hi for t in tokens),
        "metrics": {
            "size_in_range": round(in_range, 3),
            "coverage": round(coverage, 3),
            "structure_integrity": round(structure, 3),
            "uniqueness": round(uniqueness, 3),
            "metadata_completeness": round(completeness, 3),
        },
        "structured_content": {"code_chunks": len(code_chunks), "table_chunks": len(table_chunks),
                               "tables_extracted": sum(len(d["tables"]) for d in doc_infos),
                               "code_blocks_extracted": sum(len(d["code_blocks"]) for d in doc_infos)},
        "overall_score": round(score, 1),
    }


def print_report(report: dict) -> None:
    """Human-readable quality summary."""
    print("\n" + "=" * 56 + "\n QUALITY REPORT\n" + "=" * 56)
    print(f" Documents: {report['documents']}   Chunks: {report['chunks']}   {report['chunks_by_type']}")
    t = report["tokens"]
    print(f" Tokens/chunk -> min {t['min']}, median {t['median']}, mean {t['mean']}, max {t['max']}")
    print(f" Undersized: {report['undersized_chunks']}   Oversized: {report['oversized_chunks']}")
    for name, val in report["metrics"].items():
        bar = "#" * int(val * 20)
        print(f" {name:<22} {val:>5.3f} |{bar:<20}|")
    print(f" Structured content: {report['structured_content']}")
    print(f"\n OVERALL SCORE: {report['overall_score']}/100\n" + "=" * 56)


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================
def run_pipeline(input_dir: Path, output_dir: Path, cfg: Config) -> dict:
    """Run all steps and write results to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    docs = ingest(input_dir)                                   # STEP 1
    if not docs:
        raise SystemExit(f"No supported files found in {input_dir}")

    all_chunks: List[Chunk] = []
    doc_infos: List[dict] = []
    doc_blocks: Dict[str, List[Block]] = {}

    for doc in docs:
        front: dict = {}
        if doc.doc_type == "markdown":                         # STEP 2: parse by type
            blocks, front = parse_markdown(doc.raw_text)
        elif doc.doc_type == "log":
            blocks = parse_log(doc.raw_text)
        else:
            blocks = parse_text(doc.raw_text)

        blocks = clean_blocks(blocks)                          # STEP 4: clean
        info = extract_document_data(doc, blocks, front)       # STEP 3: extract structure
        info["clean_chars"] = sum(len(b.text) for b in blocks)

        pieces = (chunk_logs(blocks, cfg) if doc.doc_type == "log"   # STEP 5: chunk
                  else chunk_blocks(blocks, cfg))
        chunks = build_chunks(info, front, pieces)             # STEP 6: metadata

        info["chunk_count"] = len(chunks)
        doc_infos.append(info)
        doc_blocks[doc.doc_id] = blocks
        all_chunks.extend(chunks)
        print(f" processed {doc.path:<28} type={doc.doc_type:<9} blocks={len(blocks):<4} chunks={len(chunks)}")

    add_keywords(all_chunks, cfg)                              # STEP 6b: corpus-level keywords
    index = build_index(all_chunks)                            # STEP 7: index
    report = compute_quality(all_chunks, doc_blocks, doc_infos, cfg)   # STEP 8: quality

    # Write outputs
    with open(output_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    (output_dir / "documents.json").write_text(json.dumps(doc_infos, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print_report(report)
    print(f" Output written to: {output_dir.resolve()}")
    return report


def load_for_search(output_dir: Path) -> Tuple[dict, Dict[str, dict]]:
    """Load a previously built index and chunk store from disk."""
    index = json.loads((output_dir / "index.json").read_text(encoding="utf-8"))
    chunks = {}
    with open(output_dir / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return index, chunks


# =============================================================================
# DEMO DATA (so you can try the system without preparing files)
# =============================================================================
DEMO_MD = """---
title: Farmer Platform Setup Guide
tags: [setup, api, farmers]
author: Team
---
# Farmer Platform Setup Guide

This guide explains how to install and run the coordination platform for smallholder farmers.
It covers the API server, the SMS gateway and the database. Contact support@example.org for help.

## Installation

Install the dependencies first. The platform needs Python 3.10 or newer and a PostgreSQL database.
Then start the server using the commands below.

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

## Crop price table

| Crop    | Region    | Price (KES/kg) |
|---------|-----------|----------------|
| Maize   | Nyandarua | 52             |
| Potato  | Nyandarua | 38             |
| Cabbage | Nakuru    | 25             |

See the [market API docs](https://example.org/docs/market) for live prices.

## Troubleshooting

- If the database connection times out, check the host and port in the settings file.
- If SMS messages are delayed, restart the gateway service.
- Check logs at 192.168.1.20 for details. Last reviewed on 2026-09-01.
"""

DEMO_TXT = """FIELD NOTES

Farmers in the highlands reported late rains this season. Potato planting was delayed by about two
weeks, which may reduce yields. Cooperative leaders asked for earlier price alerts and clearer
advice on storage, because losses after harvest remain high.

MARKET OBSERVATIONS

Prices for cabbage dropped sharply after a surplus arrived from nearby counties. Traders said that
transport costs are the biggest barrier for small producers who want to reach larger towns.
"""

DEMO_LOG = """2026-09-01 10:00:01,120 INFO [api] Server started on port 8000
2026-09-01 10:00:05,300 INFO [sms] Gateway connected
2026-09-01 10:03:10,000 WARN [db] Slow query detected (2.3s)
2026-09-01 10:05:44,500 ERROR [db] Connection timeout while contacting 10.0.0.5:5432
Traceback (most recent call last):
  File "db.py", line 42, in connect
    raise TimeoutError("database unreachable")
TimeoutError: database unreachable
2026-09-01 10:05:45,000 ERROR [sms] Message delivery failed
2026-09-01 10:05:46,000 ERROR [sms] Message delivery failed
2026-09-01 10:05:47,000 ERROR [sms] Message delivery failed
2026-09-01 10:06:00,000 INFO [db] Connection restored
"""


def write_demo_files(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "setup_guide.md").write_text(DEMO_MD, encoding="utf-8")
    (folder / "field_notes.txt").write_text(DEMO_TXT, encoding="utf-8")
    (folder / "server.log").write_text(DEMO_LOG, encoding="utf-8")


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Document preprocessor for a RAG knowledge base")
    ap.add_argument("--input", type=Path, help="folder containing .txt/.md/.log files")
    ap.add_argument("--output", type=Path, default=Path("kb_output"), help="output folder")
    ap.add_argument("--demo", action="store_true", help="generate sample documents and process them")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--min-tokens", type=int, default=30)
    ap.add_argument("--overlap", type=int, default=1, help="overlap sentences for split paragraphs")
    ap.add_argument("--search", type=str, help="query the existing index in --output")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--type", dest="doc_type", choices=["markdown", "text", "log"], help="filter by type")
    ap.add_argument("--has-code", action="store_true", help="only return chunks containing code")
    ap.add_argument("--level", type=str, help="only log chunks containing this level (e.g. ERROR)")
    args = ap.parse_args()

    cfg = Config(max_tokens=args.max_tokens, min_tokens=args.min_tokens, overlap_sentences=args.overlap)

    # Build (or rebuild) the knowledge base when asked
    if args.demo:
        demo_dir = args.output / "demo_docs"
        write_demo_files(demo_dir)
        run_pipeline(demo_dir, args.output, cfg)
    elif args.input:
        run_pipeline(args.input, args.output, cfg)

    # Query the knowledge base
    query = args.search or ("database timeout" if args.demo else None)
    if query:
        index, chunks = load_for_search(args.output)
        hits = search(index, chunks, query, args.top_k, args.doc_type, args.has_code, args.level)
        print(f"\n Search: '{query}'  ({len(hits)} result(s))")
        for rank, (score, c) in enumerate(hits, 1):
            preview = c["text"].replace("\n", " ")[:110]
            print(f"  {rank}. score={score:.2f} [{c['doc_type']}] {c['source']} :: {c['section'] or '-'}")
            print(f"     keywords={c['keywords']}")
            print(f"     {preview}...")
    elif not (args.demo or args.input):
        ap.print_help()


if __name__ == "__main__":
    main()