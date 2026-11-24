"""MkDocs hooks: make GitHub/Notion-flavoured notes render correctly with Python-Markdown.

The notes under docs/ are written and previewed as GFM (GitHub, VS Code,
Notion import). Python-Markdown, which MkDocs uses, differs in a few ways that
silently break pages. Everything here runs on the in-memory text at build time;
the source .md files are never modified.

* Nested lists: GFM nests by the parent's content column (2 or 3 spaces);
  Python-Markdown needs 4 spaces per level.            -> normalize_lists
* Lists, tables and ``$$`` blocks directly after a paragraph line are not
  recognized without a blank line.                       -> blank lines inserted
* ``|`` inside ``$...$`` in a table row splits the cell (``\\|`` is GFM's
  escape for a literal pipe there).                      -> ``\\vert``
* Links to repo files outside docs/ (AGENTS.md, scripts) would 404 on the
  site.                                                  -> GitHub blob URLs
* Footnote definitions that are never cited (author notes) render as plain
  paragraphs instead of footnotes with a dangling back-link.
* Images no page references are dropped from the build to keep the published
  site small.                                            -> on_files
* Section titles come from a short map instead of capitalized folder names.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

log = logging.getLogger("mkdocs.hooks.ai-learn")

# --------------------------------------------------------------------------- titles

SECTION_TITLES = {
    "embedding": "Embedding",
    "llm": "LLM",
    "post-training": "后训练",
    "llm/long-context": "长上下文",
    "post-training/peft": "PEFT 微调",
    "post-training/distillation": "知识蒸馏",
}

# Per-paper files share a name; label them by role in the sidebar
PAGE_TITLES = {"zh": "中英对照", "en": "英文原文"}

# --------------------------------------------------------------------------- regexes

FENCE_RE = re.compile(r"^(?P<ind> *)(?P<fence>`{3,}|~{3,})")
LIST_RE = re.compile(r"^(?P<ind> *)(?P<mark>[-*+]|\d{1,9}\.)(?P<sp> +|$)")
TABLE_SEP_RE = re.compile(r"^ *\|? *:?-{2,}:? *(\| *:?-{2,}:? *)*\|? *$")
INLINE_MATH_RE = re.compile(r"(?<![\\$])\$(?!\$)(?P<body>(?:\\.|[^$\\])+?)(?<!\\)\$(?!\$)")
LINK_RE = re.compile(r"(\]\()([^)\s]+)((?:\s+\"[^\"]*\")?\))")
IMG_SRC_RE = re.compile(r"""(?:src|href)=["']([^"']+)["']""")
ASSET_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp"}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _blockish(line: str) -> bool:
    """Lines after which a list/table/math block needs no separating blank line."""
    s = line.lstrip()
    return (not s) or s.startswith(("#", "|", ">", "<", "$$", "```", "~~~")) or bool(LIST_RE.match(line))


@dataclass
class _Item:
    content: int   # source content column (GFM)
    new_ind: int   # normalized indentation of the marker


# --------------------------------------------------------------------------- markdown pass


def _fix_table_math(line: str) -> str:
    """Inside a table row, ``|`` in inline math would split the cell."""
    if "$" not in line or "|" not in line:
        return line

    def repl(m: re.Match) -> str:
        body = m.group("body")
        if "|" not in body:
            return m.group(0)
        body = body.replace("\\|", r"\vert ").replace("|", r"\vert ")
        return f"${body}$"

    return INLINE_MATH_RE.sub(repl, line)


FOOTDEF_RE = re.compile(r"^\[\^([^\]]+)\]:[ \t]?", re.M)


def _orphan_footnotes(md: str) -> str:
    """Footnote definitions nobody cites (author notes in translated papers) become
    plain paragraphs; otherwise Python-Markdown emits a back-link to a missing anchor."""
    labels = set(FOOTDEF_RE.findall(md))
    if not labels:
        return md
    cited = {lab for lab in labels if re.search(r"\[\^" + re.escape(lab) + r"\](?!:)", md)}
    orphans = labels - cited
    if not orphans:
        return md
    return FOOTDEF_RE.sub(lambda m: "" if m.group(1) in orphans else m.group(0), md)


def normalize(md: str) -> str:
    md = _orphan_footnotes(md)
    src = md.expandtabs(4).split("\n")
    out: list[str] = []
    stack: list[_Item] = []
    fence: str | None = None      # active fence marker
    block_delta = 0               # reindent applied to lines of the active fence / math block
    in_math = False
    blank_pending = False         # emit a blank line before the next non-blank line

    def shift(line: str, delta: int) -> str:
        if delta > 0:
            return " " * delta + line
        if delta < 0:
            cut = min(-delta, _indent(line))
            return line[cut:]
        return line

    def prev() -> str:
        return out[-1] if out else ""

    def ensure_blank() -> None:
        if out and out[-1].strip():
            out.append("")

    def reindent_continuation(line: str, ind: int) -> tuple[str, int]:
        """Map a non-marker line inside a list to Python-Markdown indentation."""
        k = len(stack) - 1
        while k >= 0 and ind < stack[k].content:
            k -= 1
        if k < 0:
            return line, 0
        del stack[k + 1:]
        new = stack[k].new_ind + 4 + (ind - stack[k].content)
        return " " * new + line[ind:], new - ind

    for i, line in enumerate(src):
        stripped = line.strip()

        # ---- inside fenced code: only re-indent along with the enclosing list
        if fence is not None:
            out.append(shift(line, block_delta))
            m = FENCE_RE.match(line)
            if m and m.group("fence")[0] == fence[0] and len(m.group("fence")) >= len(fence) \
                    and not line[m.end():].strip():
                fence = None
                block_delta = 0
            continue

        # ---- inside $$ math: keep verbatim (re-indented), close on a line ending with $$.
        # A blank line would end the paragraph and split the formula, so drop it.
        if in_math:
            if not stripped:
                continue
            out.append(shift(line, block_delta))
            if stripped.endswith("$$"):
                in_math = False
                block_delta = 0
                blank_pending = True
            continue

        if not stripped:
            out.append(line)
            blank_pending = False
            continue

        if blank_pending:
            if not (_is_table_row(line) and _is_table_row(prev())):
                ensure_blank()
            blank_pending = False

        ind = _indent(line)
        is_quote = line.lstrip().startswith(">")

        # ---- list bookkeeping & re-indentation (blockquotes left alone)
        m = LIST_RE.match(line) if not is_quote else None
        delta = 0
        if m:
            while stack and ind < stack[-1].content:
                stack.pop()
            if not stack and prev().strip() and not _blockish(prev()):
                ensure_blank()
            new_ind = 4 * len(stack)
            sp = m.group("sp")
            content = ind + len(m.group("mark")) + (len(sp) if 0 < len(sp) <= 4 else 1)
            stack.append(_Item(content=content, new_ind=new_ind))
            delta = new_ind - ind
            line = " " * new_ind + line[ind:]
        elif stack:
            if ind == 0:
                if not prev().strip() or line.lstrip().startswith(("#", "|", ">", "$$", "```", "~~~", "---")):
                    stack.clear()
            else:
                line, delta = reindent_continuation(line, ind)
                if delta == 0 and not prev().strip() and ind < (stack[0].content if stack else 0):
                    stack.clear()

        s = line.lstrip()

        # ---- fenced code opener
        fm = FENCE_RE.match(line)
        if fm:
            fence = fm.group("fence")
            block_delta = delta
            out.append(line)
            continue

        # ---- $$ display math
        if s.startswith("$$"):
            if prev().strip() and not is_quote:
                ensure_blank()
            out.append(line)
            body = s[2:]
            if body.rstrip().endswith("$$") and len(body.strip()) >= 2:
                blank_pending = True          # one-line $$ ... $$
            else:
                in_math = True
                block_delta = delta
            continue

        # ---- tables
        if _is_table_row(line):
            nxt = src[i + 1] if i + 1 < len(src) else ""
            if TABLE_SEP_RE.match(nxt) and prev().strip() and not _is_table_row(prev()):
                ensure_blank()
            line = _fix_table_math(line)
            out.append(line)
            nxt_s = nxt.strip()
            if nxt_s and not nxt_s.startswith("|"):
                blank_pending = True
            continue

        out.append(line)

    return "\n".join(out)


# --------------------------------------------------------------------------- links


def _rewrite_outside_links(md: str, page, config, files) -> str:
    """Links to repo files that are not part of the site -> GitHub URLs.

    Covers files outside docs/ (AGENTS.md, scripts/) and files inside docs/
    that ``exclude_docs`` keeps out of the build (PDFs, the writing guide).
    """
    docs_dir = Path(config["docs_dir"]).resolve()
    repo_root = Path(config["config_file_path"]).resolve().parent
    repo_url = (config.get("repo_url") or "").rstrip("/")
    if not repo_url:
        return md
    base = Path(page.file.abs_src_path).parent

    def repl(m: re.Match) -> str:
        target = m.group(2)
        if re.match(r"^([a-z][a-z0-9+.-]*:|#|/)", target, re.I):
            return m.group(0)
        path, _, anchor = target.partition("#")
        abs_t = Path(os.path.normpath(base / unquote(path)))
        try:
            in_docs = abs_t.relative_to(docs_dir).as_posix()
        except ValueError:
            in_docs = None
        f = files.get_file_from_path(in_docs) if in_docs is not None else None
        published = f is not None and not f.inclusion.is_excluded()
        if in_docs is not None and (not abs_t.exists() or abs_t.is_dir() or published):
            return m.group(0)  # part of the site (or genuinely missing: let validation report it)
        try:
            rel = abs_t.relative_to(repo_root).as_posix()
        except ValueError:
            return m.group(0)
        kind = "tree" if abs_t.is_dir() else "blob"
        url = f"{repo_url}/{kind}/main/{rel}" + (f"#{anchor}" if anchor else "")
        return f"{m.group(1)}{url}{m.group(3)}"

    return LINK_RE.sub(repl, md)


# --------------------------------------------------------------------------- hooks


def on_page_markdown(markdown: str, page, config, files):
    markdown = _rewrite_outside_links(markdown, page, config, files)
    return normalize(markdown)


def on_files(files, config):
    """Drop images under docs/ that no page references."""
    docs_dir = Path(config["docs_dir"]).resolve()
    referenced: set[str] = set()
    for f in files.documentation_pages():
        text = Path(f.abs_src_path).read_text(encoding="utf-8")
        base = Path(f.abs_src_path).parent
        targets = [m.group(2) for m in LINK_RE.finditer(text)] + IMG_SRC_RE.findall(text)
        for t in targets:
            if re.match(r"^([a-z][a-z0-9+.-]*:|#)", t, re.I):
                continue
            referenced.add(os.path.normpath(base / unquote(t.partition("#")[0])))

    dropped, size = 0, 0
    for f in list(files):
        p = Path(f.abs_src_path) if f.abs_src_path else None
        if p is None or p.suffix.lower() not in ASSET_EXT:
            continue
        try:
            p.resolve().relative_to(docs_dir)
        except ValueError:
            continue  # theme / custom_dir assets
        if os.path.normpath(p) not in referenced:
            files.remove(f)
            dropped += 1
            size += p.stat().st_size
    log.info("ai-learn: dropped %d unreferenced images (%.0f MB)", dropped, size / 2**20)
    return files


def on_nav(nav, config, files):
    docs_dir = Path(config["docs_dir"]).resolve()

    def walk(items):
        for it in items:
            if getattr(it, "is_page", False) and it.file and it.file.name in PAGE_TITLES:
                paper = Path(it.file.src_uri).parent.name
                it.title = f"{paper} · {PAGE_TITLES[it.file.name]}"
            if getattr(it, "is_section", False):
                key = _section_dir(it, docs_dir)
                if key and key in SECTION_TITLES:
                    it.title = SECTION_TITLES[key]
                elif key:
                    it.title = key.rsplit("/", 1)[-1]  # keep folder name as written, no auto-capitalize
                walk(it.children)

    walk(nav.items)
    return nav


def _section_dir(section, docs_dir: Path) -> str | None:
    """docs-relative directory of a nav section, from the first page inside it."""
    queue = list(section.children)
    while queue:
        it = queue.pop(0)
        if getattr(it, "is_page", False) and it.file:
            parts = Path(it.file.src_uri).parent.parts
            d = _depth(section)
            return "/".join(parts[:d]) if len(parts) >= d else None
        if getattr(it, "is_section", False):
            queue.extend(it.children)
    return None


def _depth(section) -> int:
    d = 1
    p = section.parent
    while p is not None:
        d += 1
        p = p.parent
    return d
