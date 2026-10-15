#!/usr/bin/env python3
"""Crop paper tables out of a PDF into <folder>/tables/tableNN.png.

Why: a markdown table loses the typography of the original (booktabs rules,
stacked headers, aligned decimals, sub/superscripts, bolded best values).
Cropping the PDF gives a rendering pixel-identical to the published paper, and
needs no browser and no LaTeX install -- only poppler, which this repo already
relies on for figure fallback.

How it works:
  1. `pdftotext -bbox-layout` gives <block> paragraphs; the block starting with
     "Table N:" is the caption, which pins down the page and the exact caption
     box (captions wrap over several lines, so the block beats a word match).
  2. That page is rendered once with `pdftoppm` at --dpi.
  3. The table's horizontal rules (\\toprule / \\midrule / \\hline ...) are found
     as long runs of dark pixels. Rules sharing an x-span cluster into one
     table. Rules are unambiguous, unlike text gaps, and this works whether the
     table sits in one column or spans the page.
  4. The cluster nearest the caption wins -- searched on *both* sides, since
     ACM-style papers put the caption above the table and ACL-style papers put
     it below.
  5. Crop = union of caption block and rule cluster, then trim to the ink.

Verified against both layouts: Margin-MSE (ACM, caption above, booktabs) and
Multilingual-SBERT (ACL, caption below, per-row rules).

Usage:
    python3 _crop_pdf_tables.py <paper-folder> [...] [--dpi 400]
    python3 _crop_pdf_tables.py Margin-MSE Multilingual-SBERT
    python3 _crop_pdf_tables.py Margin-MSE --tables 2,3 --no-caption
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PAPERS_DIR = Path(__file__).resolve().parent

BLOCK_RE = re.compile(
    r'<block xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</block>',
    re.S,
)
WORD_RE = re.compile(r">([^<]*)</word>")


# ---------------------------------------------------------------- text layer


def caption_blocks(pdf: Path) -> dict[int, tuple[int, float, float, float, float]]:
    """Map table number -> (page_no, x0, y0, x1, y1) of its caption block."""
    out = subprocess.run(
        ["pdftotext", "-bbox-layout", str(pdf), "-"], capture_output=True, check=True
    ).stdout.decode("utf-8", errors="ignore")

    caps: dict[int, tuple[int, float, float, float, float]] = {}
    for pno, pm in enumerate(re.finditer(r"<page\b.*?>(.*?)</page>", out, re.S), 1):
        for bm in BLOCK_RE.finditer(pm.group(1)):
            words = WORD_RE.findall(bm.group(5))
            if len(words) < 2 or words[0] != "Table":
                continue
            # "Table 7:" (ACM/ACL) or "Table 7." (CVPR/NeurIPS). Requiring the
            # block to *start* here keeps body references out: "see Table 7."
            # is mid-block, and "Table 7 shows" has "7" as the second word.
            m = re.fullmatch(r"(\d+)[:.]", words[1])
            if not m:
                continue
            num = int(m.group(1))
            if num in caps:  # keep the first occurrence
                continue
            caps[num] = (
                pno, float(bm.group(1)), float(bm.group(2)),
                float(bm.group(3)), float(bm.group(4)),
            )
    return caps


# -------------------------------------------------------------- pixel layer


def render_page(pdf: Path, page: int, dpi: int):
    from PIL import Image

    stem = Path("/tmp") / f"_tblcrop_p{page}"
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-f", str(page), "-l", str(page), "-png",
         str(pdf), str(stem)],
        check=True, capture_output=True,
    )
    hits = sorted(stem.parent.glob(stem.name + "*.png"))
    if not hits:
        raise RuntimeError(f"pdftoppm produced nothing for page {page}")
    img = Image.open(hits[0]).convert("RGB")
    img.load()
    for h in hits:
        h.unlink(missing_ok=True)
    return img


def horizontal_rules(img, min_frac: float, dark: int = 160):
    """Horizontal rules as (y, x_start, x_end), from long runs of dark pixels."""
    from PIL import ImageOps

    gray = ImageOps.grayscale(img)
    w, h = gray.size
    px = gray.load()
    min_run = int(w * min_frac)

    rules = []
    for y in range(h):
        run = best = 0
        start = best_start = 0
        for x in range(w):
            if px[x, y] < dark:
                if run == 0:
                    start = x
                run += 1
                if run > best:
                    best, best_start = run, start
            else:
                run = 0
        if best >= min_run:
            rules.append((y, best_start, best_start + best))
    return rules


def cluster_rules(rules, y_tol: float, x_tol: float):
    """Group rules sharing an x-span into (y_top, y_bot, x_start, x_end, count)."""
    collapsed = []
    for y, xa, xb in rules:
        if collapsed and y - collapsed[-1][0] <= 2 and abs(xa - collapsed[-1][1]) <= x_tol:
            continue  # same drawn rule, next scanline
        collapsed.append((y, xa, xb))

    groups: list[list[tuple[int, int, int]]] = []
    for r in collapsed:
        for g in groups:
            if (abs(g[-1][1] - r[1]) <= x_tol and abs(g[-1][2] - r[2]) <= x_tol
                    and r[0] - g[-1][0] <= y_tol):
                g.append(r)
                break
        else:
            groups.append([r])

    return [
        (min(r[0] for r in g), max(r[0] for r in g),
         min(r[1] for r in g), max(r[2] for r in g), len(g))
        for g in groups
    ]


def trim(img, bg: int = 245):
    from PIL import ImageOps

    gray = ImageOps.grayscale(img)
    box = gray.point(lambda v: 255 if v < bg else 0).getbbox()
    return img.crop(box) if box else img


# ------------------------------------------------------------------- driver


def gap_to_caption(cluster, cap_y0: float, cap_y1: float) -> float:
    """Vertical gap between a rule cluster and the caption, either side."""
    y_top, y_bot = cluster[0], cluster[1]
    if y_bot <= cap_y0:
        return cap_y0 - y_bot          # table above the caption (ACL style)
    if y_top >= cap_y1:
        return y_top - cap_y1          # table below the caption (ACM style)
    return 0.0                          # overlapping


def caption_slots(caps: dict, pno: int, num: int, page_h_px: float, scale: float
                  ) -> dict[str, tuple[float, float]]:
    """The two y-ranges (px) that could hold table `num`'s rules.

    Captions partition a page, so a table body lives in the gap either directly
    above or directly below its own caption, never across another caption. This
    keeps the rule grouping free to be generous (booktabs midrules can sit
    200pt apart) without ever merging two stacked tables.
    """
    _pno, cx0, cy0, cx1, cy1 = caps[num]
    cx0, cy0, cx1, cy1 = cx0 * scale, cy0 * scale, cx1 * scale, cy1 * scale

    # Only captions in the same column may bound this one. In a two-column
    # layout a caption in the other column sits at an unrelated y and would
    # otherwise clip the table (it truncated Table 5 of the SBERT paper).
    others = [
        (c[2] * scale, c[4] * scale)
        for n, c in caps.items()
        if c[0] == pno and n != num
        and min(c[3] * scale, cx1) - max(c[1] * scale, cx0) > 0
    ]
    prev_end = max([y1 for _y0, y1 in others if y1 <= cy0] or [0.0])
    next_start = min([y0 for y0, _y1 in others if y0 >= cy1] or [page_h_px])
    return {"above": (prev_end, cy0), "below": (cy1, next_start)}


def crop_tables(folder: Path, dpi: int, wanted, pad_pt: float,
                with_caption: bool, min_frac: float, max_gap_pt: float) -> int:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"  no PDF in {folder}")
        return 0
    pdf = pdfs[0]
    print(f"== {folder.name} :: {pdf.name}")

    caps = caption_blocks(pdf)
    if not caps:
        print("  no table captions found")
        return 0

    out_dir = folder / "tables"
    out_dir.mkdir(exist_ok=True)
    scale = dpi / 72.0
    pad = pad_pt * scale

    rendered: dict[int, object] = {}
    page_rules: dict[int, list] = {}

    def best_in(num: int, slot: str):
        """Best rule cluster for table `num` in its 'above'/'below' slot."""
        pno, cx0, cy0, cx1, cy1 = caps[num]
        if pno not in rendered:
            rendered[pno] = render_page(pdf, pno, dpi)
            page_rules[pno] = horizontal_rules(rendered[pno], min_frac)
        img = rendered[pno]

        lo, hi = caption_slots(caps, pno, num, img.height, scale)[slot]
        band = [r for r in page_rules[pno] if lo <= r[0] <= hi]
        groups = cluster_rules(band, y_tol=320 * scale, x_tol=14 * scale)
        cands = [
            g for g in groups
            if g[4] >= 2
            and g[3] > cx0 * scale - 30 * scale
            and g[2] < cx1 * scale + 30 * scale
            and gap_to_caption(g, cy0 * scale, cy1 * scale) <= max_gap_pt * scale
        ]
        if not cands:
            return None
        return min(cands, key=lambda g: gap_to_caption(g, cy0 * scale, cy1 * scale))

    todo = [n for n in sorted(caps) if not wanted or n in wanted]

    # Papers are internally consistent: ACM puts the caption above the table,
    # ACL below. Decide which by voting on the tables where only one slot has
    # rules at all, so the ambiguous ones inherit the paper's own convention.
    options = {n: {s: best_in(n, s) for s in ("above", "below")} for n in todo}
    votes = {"above": 0, "below": 0}
    for opt in options.values():
        sides = [s for s, g in opt.items() if g]
        if len(sides) == 1:
            votes[sides[0]] += 1
    house_side = "above" if votes["above"] >= votes["below"] else "below"
    print(f"  layout: tables usually sit {house_side} the caption "
          f"(unambiguous votes above={votes['above']} below={votes['below']})")

    tie = 8 * scale
    found = 0
    for num in todo:
        pno, cx0, cy0, cx1, cy1 = caps[num]
        opt = options[num]

        # A table hugs its own caption, so the nearer cluster is normally the
        # right one. Only when the two sides are within `tie` -- e.g. two
        # tables stacked in one column -- fall back to the paper's own habit.
        # Papers can be mixed (InPars: body captions below, appendix above),
        # so this is decided per table rather than globally.
        if opt["above"] and opt["below"]:
            ga = gap_to_caption(opt["above"], cy0 * scale, cy1 * scale)
            gb = gap_to_caption(opt["below"], cy0 * scale, cy1 * scale)
            if abs(ga - gb) > tie:
                side = "above" if ga < gb else "below"
            else:
                side = house_side
        else:
            side = "above" if opt["above"] else "below"

        pick = opt[side]
        if not pick:
            print(f"  Table {num}: p{pno} no rule cluster near caption, skipped")
            continue
        y_top, y_bot, x_a, x_b, nrule = pick
        img = rendered[pno]

        if with_caption:
            top = min(y_top, cy0 * scale)
            bot = max(y_bot, cy1 * scale)
            left = min(x_a, cx0 * scale)
            right = max(x_b, cx1 * scale)
        else:
            top, bot, left, right = y_top, y_bot, x_a, x_b

        box = (
            max(0, int(left - pad)), max(0, int(top - pad)),
            min(img.width, int(right + pad)), min(img.height, int(bot + pad)),
        )
        out = trim(img.crop(box))
        dest = out_dir / f"table{num:02d}.png"
        out.save(dest)
        print(f"  Table {num}: p{pno} rules={nrule} {side:5s} "
              f"y={y_top/scale:.0f}-{y_bot/scale:.0f}pt "
              f"-> {dest.name} {out.width}x{out.height}")
        found += 1

    print(f"  done: {found}/{len(wanted or caps)} tables at {dpi} dpi -> {out_dir}")
    return found


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("folders", nargs="+", help="paper folder(s) under this directory")
    ap.add_argument("--dpi", type=int, default=400, help="render DPI (default 400)")
    ap.add_argument("--tables", help="comma list of table numbers (default: all)")
    ap.add_argument("--pad", type=float, default=5.0,
                    help="padding in PDF points (default 5)")
    ap.add_argument("--no-caption", action="store_true",
                    help="crop the tabular only, dropping the caption")
    ap.add_argument("--min-frac", type=float, default=0.18,
                    help="min rule width as a fraction of page width (default 0.18)")
    ap.add_argument("--max-gap", type=float, default=40.0,
                    help="max caption-to-table gap in points (default 40)")
    args = ap.parse_args(argv[1:])

    wanted = {int(t) for t in args.tables.split(",")} if args.tables else None
    for name in args.folders:
        folder = Path(name) if Path(name).is_dir() else PAPERS_DIR / name
        if not folder.is_dir():
            print(f"  skip, not a directory: {name}")
            continue
        crop_tables(folder, args.dpi, wanted, args.pad,
                    not args.no_caption, args.min_frac, args.max_gap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
