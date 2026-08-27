#!/usr/bin/env python3
"""Extract figures from paper PDFs and embed into *_zh.md translations."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import fitz

PAPERS_DIR = Path(__file__).resolve().parent

# (pdf_stem, zh_md_name) — zh may share stem with pdf
PAPER_PAIRS = [
    "ANCE_2007.00808",
    "Conan-embedding-v1_2408.15710",
    "Conan-embedding-v2_2509.12892",
    "NV-Retriever_2407.15831",
    "LLM-DA-TPR_2405.11971",
    "DeVE-QA_2409.04388",
    "RocketQA_2010.08191",
    "Token-Prepending_2412.11556",
]

# Prefer true captions: "Figure 1:" / "Figure 1." at line start (not in-body refs)
CAPTION_RE = re.compile(
    # also accept '|' delimiter (used e.g. by Google's SigLIP2 style)
    r"(?m)^(?:\s*)(?:Figure|Fig\.?|FIGURE)\s+(\d+)\s*[:.：|]",
)


def union_rects(rects: list[fitz.Rect]) -> fitz.Rect | None:
    if not rects:
        return None
    u = fitz.Rect(rects[0])
    for r in rects[1:]:
        u |= r
    return u


def figure_clip(page: fitz.Page, caption: fitz.Rect) -> fitz.Rect:
    """Crop region likely containing the figure above its caption."""
    page_rect = page.rect
    candidates: list[fitz.Rect] = []

    # Raster images
    for info in page.get_image_info(xrefs=True):
        r = fitz.Rect(info["bbox"])
        if r.y1 <= caption.y0 + 8 and r.get_area() > 2000:
            candidates.append(r)

    # Vector drawings
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.y1 <= caption.y0 + 8 and r.get_area() > 800:
            candidates.append(r)

    clip = union_rects(candidates)
    if clip is None or clip.get_area() < 5000:
        # Fallback: band above caption (skip header strip)
        top = max(page_rect.y0 + 36, caption.y0 - page_rect.height * 0.55)
        clip = fitz.Rect(page_rect.x0 + 24, top, page_rect.x1 - 24, caption.y0 - 2)
    else:
        # Pad and extend down to caption
        clip = fitz.Rect(
            max(page_rect.x0 + 12, clip.x0 - 8),
            max(page_rect.y0 + 24, clip.y0 - 8),
            min(page_rect.x1 - 12, clip.x1 + 8),
            min(caption.y0 - 1, clip.y1 + 4),
        )

    # Keep sane size
    if clip.height < 40 or clip.width < 40:
        clip = fitz.Rect(
            page_rect.x0 + 24,
            max(page_rect.y0 + 36, caption.y0 - page_rect.height * 0.5),
            page_rect.x1 - 24,
            caption.y0 - 2,
        )
    return clip


def collect_captions(doc: fitz.Document) -> list[tuple[int, int, fitz.Rect]]:
    """List of (fig_num, page_idx, caption_rect), reading order."""
    items: list[tuple[int, int, fitz.Rect]] = []
    seen: set[tuple[int, int]] = set()
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for m in CAPTION_RE.finditer(page.get_text()):
            fig_num = int(m.group(1))
            key = (page_idx, fig_num)
            if key in seen:
                continue
            hits = page.search_for(m.group(0))
            if not hits:
                # try shorter query
                hits = page.search_for(f"Figure {fig_num}") or page.search_for(
                    f"Fig. {fig_num}"
                )
            if not hits:
                continue
            caption = sorted(hits, key=lambda r: (r.y0, r.x0))[0]
            seen.add(key)
            items.append((fig_num, page_idx, caption))
    items.sort(key=lambda t: (t[1], t[2].y0, t[0]))
    return items


def extract_figures(pdf_path: Path, out_dir: Path, dpi: float = 160) -> dict[int, str]:
    """Return map fig_num -> relative path from papers dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    captions = collect_captions(doc)
    # Prefer first occurrence of each figure number in reading order
    chosen: dict[int, tuple[int, fitz.Rect, float | None]] = {}
    for i, (fig_num, page_idx, caption) in enumerate(captions):
        if fig_num in chosen:
            continue
        # Upper bound: previous caption on same page (multi-fig pages)
        prev_y: float | None = None
        for j in range(i - 1, -1, -1):
            if captions[j][1] != page_idx:
                break
            prev_y = captions[j][2].y1
            break
        chosen[fig_num] = (page_idx, caption, prev_y)

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    rel_map: dict[int, str] = {}
    for fig_num in sorted(chosen):
        page_idx, caption, prev_y = chosen[fig_num]
        page = doc[page_idx]
        clip = figure_clip(page, caption)
        if prev_y is not None and clip.y0 < prev_y:
            clip.y0 = min(prev_y + 2, caption.y0 - 4)
        # Intersect with page and ensure positive area
        clip &= page.rect
        if clip.is_empty or clip.height < 24 or clip.width < 24:
            top = (prev_y + 2) if prev_y is not None else max(page.rect.y0 + 36, caption.y0 - 280)
            clip = fitz.Rect(
                page.rect.x0 + 24,
                min(top, caption.y0 - 40),
                page.rect.x1 - 24,
                caption.y0 - 2,
            )
            clip &= page.rect
        if clip.is_empty or clip.height < 8 or clip.width < 8:
            print(f"  fig {fig_num}: SKIP invalid clip on page {page_idx + 1}")
            continue
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        if pix.width < 8 or pix.height < 8:
            print(f"  fig {fig_num}: SKIP empty pixmap on page {page_idx + 1}")
            continue
        fname = f"fig{fig_num:02d}.png"
        out_path = out_dir / fname
        pix.save(out_path.as_posix())
        rel_map[fig_num] = f"{out_dir.name}/{fname}"
        print(f"  fig {fig_num}: page {page_idx + 1} -> {out_path.name} ({pix.width}x{pix.height})")

    doc.close()
    return rel_map


# Match: **图 1：** / **图 1**： / **图 1.** / **图 1 说明**：
FIG_CAPTION_MD = re.compile(
    r"^(?P<prefix>\s*)(?P<body>\*{0,2}图\s*(?P<num>\d+)(?:\s*说明)?\*{0,2}\s*[：:．.、].+)$",
    re.MULTILINE,
)


def embed_into_md(md_path: Path, rel_map: dict[int, str]) -> int:
    text = md_path.read_text(encoding="utf-8")
    # Drop previously inserted figure images for idempotency
    text = re.sub(
        r"!\[图\s*\d+\]\([^)]+/fig\d+\.png\)\n*",
        "",
        text,
    )

    inserted = 0

    def repl(m: re.Match) -> str:
        nonlocal inserted
        num = int(m.group("num"))
        rel = rel_map.get(num)
        if not rel:
            return m.group(0)
        inserted += 1
        return f"{m.group('prefix')}![图 {num}]({rel})\n\n{m.group(0)}"

    new_text = FIG_CAPTION_MD.sub(repl, text)
    md_path.write_text(new_text, encoding="utf-8")
    return inserted


def process_one(stem: str) -> None:
    pdf = PAPERS_DIR / f"{stem}.pdf"
    md = PAPERS_DIR / f"{stem}_zh.md"
    if not pdf.exists():
        print(f"[skip] missing pdf: {pdf.name}")
        return
    if not md.exists():
        print(f"[skip] missing md: {md.name}")
        return
    fig_dir = PAPERS_DIR / f"{stem}_figs"
    print(f"== {stem} ==")
    rel_map = extract_figures(pdf, fig_dir)
    n = embed_into_md(md, rel_map)
    print(f"  embedded {n}/{len(rel_map)} figures into {md.name}")


def main(argv: list[str]) -> int:
    stems = argv[1:] if len(argv) > 1 else PAPER_PAIRS
    for stem in stems:
        process_one(stem)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
