#!/usr/bin/env python3
"""Fallback: crop figures out of a paper PDF into <folder>/figs/figNN.png.

Use only when arXiv HTML / ar5iv has no original figures (see
``fetch_figures.py``). Each "Figure N:" caption is located in the PDF text
layer and the raster/vector content above it is rendered at --dpi.

Existing ``figs/figNN.png`` are kept unless ``--force`` -- they are usually the
better HTML originals. With ``--embed zh.md`` an image line is inserted above
every ``**图 N：**`` caption in that markdown file (idempotent).

Usage:
    python3 scripts/papers/pdf_figures.py embedding/RocketQA
    python3 scripts/papers/pdf_figures.py embedding/RocketQA --embed zh.md
    python3 scripts/papers/pdf_figures.py embedding/RocketQA --force --dpi 200

Requires PyMuPDF (``uv run --with pymupdf python3 scripts/papers/pdf_figures.py ...``).
Folders are taken as given, or else relative to docs/.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf as fitz

from _paths import pdfs_in, rel, resolve

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


def extract_figures(pdf_path: Path, out_dir: Path, dpi: float = 160,
                    force: bool = False) -> dict[int, str]:
    """Return map fig_num -> path relative to the paper folder (``figs/figNN.png``)."""
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
        rel_map[fig_num] = f"{out_dir.name}/{fname}"
        if out_path.exists() and not force:
            print(f"  fig {fig_num}: keep existing {out_path.name}")
            continue
        pix.save(out_path.as_posix())
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


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="paper folder(s), as given or relative to docs/")
    ap.add_argument("--dpi", type=float, default=160, help="render DPI (default 160)")
    ap.add_argument("--force", action="store_true", help="overwrite existing figs/figNN.png")
    ap.add_argument("--embed", metavar="MD", help="markdown file in the folder to insert images into")
    args = ap.parse_args(argv[1:])

    rc = 0
    for name in args.folders:
        folder = resolve(name)
        pdfs = pdfs_in(folder) if folder and folder.is_dir() else []
        if not pdfs:
            print(f"  skip {name}: no PDF in folder")
            rc = 1
            continue
        print(f"== {rel(folder)} ==")
        rel_map = extract_figures(pdfs[0], folder / "figs", args.dpi, args.force)
        if args.embed:
            md = folder / args.embed
            if not md.exists():
                print(f"  skip embed: {rel(md)} missing")
                rc = 1
                continue
            n = embed_into_md(md, rel_map)
            print(f"  embedded {n}/{len(rel_map)} figures into {md.name}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
