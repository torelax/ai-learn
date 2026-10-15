#!/usr/bin/env python3
"""Shrink paper PDFs before committing them.

Why: arXiv PDFs often embed figures at absurd resolution -- DINOv3 ships
images at 1515 dpi, where screen reading needs ~150 and printing ~300. Three
papers alone (DeVE-QA, DINOv3, GME) were 45 MB of a 227 MB repo. Text in these
PDFs is vector, so downsampling the rasters leaves every character, formula
and in-figure label pixel-identical; only photographs soften.

Run this on a freshly downloaded PDF *before* the first commit. Compressing a
PDF that is already committed makes the repo bigger, not smaller: the new blob
is added while the old one stays in history. Existing large PDFs are therefore
left alone until a history rewrite handles them in one pass.

Quality levels (measured on this repo's papers):
  ebook    images to ~150 dpi. Text-centric papers are indistinguishable from
           the original. The default.
  printer  images to ~300 dpi. Use when the figures *are* the contribution
           (DINOv3/DINOv2/CLIP-style visual papers), where ebook visibly
           blurs photographic detail.

Safety: the result goes to a temp file and is only swapped in if poppler can
read it, the page count matches, it actually got smaller, and the surviving
images stay at or above --min-dpi. Re-running is a no-op, since an already
shrunk file cannot shrink again.

The guard is the embedded image resolution, not a pixel diff against the
original. A mean-pixel-difference check was tried first and thrown out: it is
dominated by resampling phase rather than quality, so it reordered the presets
depending on the render DPI (at 110 dpi /screen scored "better" than /ebook, at
170 dpi the reverse). Image DPI is exact, monotonic, and is what the presets
actually control.

Usage:
    python3 _shrink_pdf.py <pdf-or-folder> [...] [--quality printer]
    python3 _shrink_pdf.py Margin-MSE --dry-run
    python3 _shrink_pdf.py DINO/DINOv3 --quality printer
    python3 _shrink_pdf.py . --dry-run          # survey everything
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUALITY = {"screen": "/screen", "ebook": "/ebook", "printer": "/printer", "prepress": "/prepress"}
# what each preset downsamples colour images to, for reference in --help output
PRESET_DPI = {"screen": 72, "ebook": 150, "printer": 300, "prepress": 300}


def mb(p: Path) -> float:
    return p.stat().st_size / 1048576


def page_count(pdf: Path) -> int | None:
    r = subprocess.run(["pdfinfo", str(pdf)], capture_output=True)
    if r.returncode != 0:
        return None
    m = re.search(r"^Pages:\s+(\d+)", r.stdout.decode(errors="ignore"), re.M)
    return int(m.group(1)) if m else None


def image_dpi(pdf: Path) -> tuple[int, int] | None:
    """(median, max) dpi of the embedded raster images, or None if there are none."""
    out = subprocess.run(["pdfimages", "-list", str(pdf)],
                         capture_output=True).stdout.decode(errors="ignore")
    dpis: list[int] = []
    for line in out.splitlines()[2:]:
        f = line.split()
        if len(f) > 12 and f[12].isdigit():
            dpis.append(int(f[12]))
    if not dpis:
        return None
    dpis.sort()
    return dpis[len(dpis) // 2], dpis[-1]


def shrink(pdf: Path, quality: str, dry_run: bool, min_dpi: int, min_mb: float) -> float:
    """Returns MB saved (0 if skipped)."""
    before = mb(pdf)
    if before < min_mb:
        return 0.0

    pages_before = page_count(pdf)
    if pages_before is None:
        print(f"  SKIP  {pdf.name}: poppler cannot read it")
        return 0.0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "out.pdf"
        subprocess.run([
            "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
            f"-dPDFSETTINGS={QUALITY[quality]}", "-dNOPAUSE", "-dQUIET", "-dBATCH",
            "-dDetectDuplicateImages=true",
            f"-sOutputFile={tmp}", str(pdf),
        ], capture_output=True)

        # gs can print errors yet still emit a usable file, so judge the output
        if not tmp.exists() or tmp.stat().st_size == 0:
            print(f"  FAIL  {pdf.name}: ghostscript produced nothing")
            return 0.0
        pages_after = page_count(tmp)
        if pages_after is None:
            print(f"  FAIL  {pdf.name}: output is not readable, kept original")
            return 0.0
        if pages_after != pages_before:
            print(f"  FAIL  {pdf.name}: pages {pages_before} -> {pages_after}, kept original")
            return 0.0
        after = mb(tmp)
        if after >= before:
            print(f"  SKIP  {pdf.name}: would grow {before:.2f} -> {after:.2f} MB")
            return 0.0

        dpi_before, dpi_after = image_dpi(pdf), image_dpi(tmp)
        note = ""
        if dpi_after is not None:
            if dpi_after[0] < min_dpi:
                print(f"  WARN  {pdf.name}: images would end at {dpi_after[0]} dpi median, "
                      f"below --min-dpi {min_dpi}. Kept original.")
                return 0.0
            b = f"{dpi_before[0]}" if dpi_before else "-"
            note = f"  img {b}->{dpi_after[0]} dpi"

        pct = (1 - after / before) * 100
        action = "WOULD" if dry_run else "OK   "
        print(f"  {action} {pdf.name}\n"
              f"        {before:6.2f} -> {after:6.2f} MB  ({pct:.0f}% smaller)"
              f"  {pages_before}p{note}")
        if not dry_run:
            shutil.copy2(tmp, pdf)
        return before - after


def collect(target: Path) -> list[Path]:
    if target.is_file() and target.suffix.lower() == ".pdf":
        return [target]
    if target.is_dir():
        return sorted(target.rglob("*.pdf"))
    return []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="PDF files or folders (relative to this dir or cwd)")
    ap.add_argument("--quality", choices=sorted(QUALITY), default="ebook",
                    help="gs preset, downsamples images to roughly "
                         + ", ".join(f"{k}={v}dpi" for k, v in PRESET_DPI.items())
                         + "; use printer when the figures are the contribution")
    ap.add_argument("--dry-run", action="store_true", help="report without replacing")
    ap.add_argument("--min-mb", type=float, default=1.0,
                    help="skip PDFs smaller than this (default 1.0)")
    ap.add_argument("--min-dpi", type=int, default=140,
                    help="refuse if surviving images fall below this median dpi. "
                         "Default 140: screen-readable, and tolerant of ghostscript "
                         "landing on 149 when it targets 150. Blocks --quality screen "
                         "(72 dpi) unless you lower it deliberately.")
    args = ap.parse_args(argv[1:])

    if not shutil.which("gs"):
        print("ghostscript (gs) not found; install it or skip compression")
        return 1

    pdfs: list[Path] = []
    for t in args.targets:
        p = Path(t)
        found = collect(p) or collect(HERE / t)
        if not found:
            print(f"  no PDF at {t}")
        pdfs += found

    print(f"quality={args.quality}  candidates>={args.min_mb}MB  min-dpi={args.min_dpi}  "
          f"{'dry-run' if args.dry_run else 'in place'}")
    saved = sum(shrink(p, args.quality, args.dry_run, args.min_dpi, args.min_mb)
                for p in pdfs)
    print(f"total {'recoverable' if args.dry_run else 'saved'}: {saved:.1f} MB "
          f"over {len(pdfs)} PDF(s) scanned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
