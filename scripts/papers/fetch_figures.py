"""Fetch a paper's original figures from arXiv HTML / ar5iv into <folder>/figs/figNN.png.

AGENTS.md prefers these over PDF crops: they are the authors' own raster/vector
exports, not re-rendered page snippets. The arXiv id is read from the PDF file
name (``<Full Title>_2401.00368.pdf``); pass ``--arxiv`` when the folder has no
PDF (e.g. a paper kept only as a link).

Figures already cropped from the PDF are moved to ``figs/_from_pdf_crop/``
(git-ignored) and used to fill numbers the HTML lacks. ``figs/.from_html``
records the source URL and makes reruns keep the HTML versions.

Usage:
    python3 scripts/papers/fetch_figures.py embedding/E5-Mistral
    python3 scripts/papers/fetch_figures.py embedding/GTE/GTE embedding/GTE/mGTE
    python3 scripts/papers/fetch_figures.py embedding/Seed --arxiv 2505.12345

Folders are taken as given, or else relative to docs/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

from _paths import arxiv_id, rel, resolve

UA = "Mozilla/5.0 (compatible; ai-learn-figure-fetch/1.0)"


def fetch(url: str, timeout: int = 60) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode() or 200, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""
    except Exception as e:
        print(f"  fetch error {url}: {e}")
        return 0, b""


def resolve_html(arxiv_id: str) -> tuple[str, str] | None:
    """Return (base_url, html_text) or None."""
    candidates = [
        # ar5iv first: its <img src="/html/<id>/assets/..."> URLs actually resolve;
        # arxiv.org's own HTML page uses relative "<id>vN/..." paths that 404 on the
        # asset host, so try ar5iv before arxiv.
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}v3",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}v2",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}v1",
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://arxiv.org/html/{arxiv_id}v1",
        f"https://arxiv.org/html/{arxiv_id}v2",
        f"https://arxiv.org/html/{arxiv_id}v3",
        f"https://arxiv.org/html/{arxiv_id}v4",
    ]
    for url in candidates:
        code, data = fetch(url)
        if code != 200 or len(data) < 12000:
            print(f"  skip {url} ({code}, {len(data)}B)")
            continue
        text = data.decode("utf-8", errors="ignore")
        if "ltx_figure" not in text and "Figure" not in text:
            print(f"  skip {url} (no figures)")
            continue
        # base for relative assets
        if "ar5iv.labs.arxiv.org" in url:
            base = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
        else:
            # strip version suffix for asset base if needed
            base = url.rstrip("/")
        print(f"  using {url}")
        return base, text
    return None


def parse_figures(html: str) -> dict[int, list[str]]:
    """Map figure number -> list of image srcs (relative or absolute)."""
    mapping: dict[int, list[str]] = {}

    # Primary: <figure class="ltx_figure"> ... Figure N[:.] ... <img src=...>
    for m in re.finditer(
        r'<figure\b[^>]*class="[^"]*ltx_figure[^"]*"[^>]*>(.*?)</figure>',
        html,
        flags=re.S | re.I,
    ):
        block = m.group(1)
        cap = re.search(
            r"(?:Figure|Fig\.?)\s*(\d+)\s*[:.：]",
            block,
            flags=re.I,
        )
        # "Figure 1</span>:"
        if not cap:
            cap = re.search(
                r"(?:Figure|Fig\.?)\s*(\d+)\s*</span>\s*[:.：]",
                block,
                flags=re.I,
            )
        if not cap:
            continue
        num = int(cap.group(1))
        imgs = re.findall(
            r'<img[^>]+src="([^"]+\.(?:png|jpg|jpeg|svg|gif))"',
            block,
            flags=re.I,
        )
        if imgs:
            mapping[num] = imgs

    # Fallback: id="S4.F1" ... <img src=...>
    for m in re.finditer(
        r'id="[^"]*F(\d+)"[^>]*>\s*<img[^>]+src="([^"]+\.(?:png|jpg|jpeg|svg|gif))"',
        html,
        flags=re.I,
    ):
        num = int(m.group(1))
        src = m.group(2)
        mapping.setdefault(num, [])
        if src not in mapping[num]:
            mapping[num].append(src)

    return mapping


def abs_url(base: str, src: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        return src
    if src.startswith("/"):
        # ar5iv: /html/ID/assets/x1.png
        if "ar5iv.labs.arxiv.org" in base:
            return "https://ar5iv.labs.arxiv.org" + src
        # arxiv html absolute path under site
        return "https://arxiv.org" + src
    return base.rstrip("/") + "/" + src.lstrip("/")


def process(folder: Path, aid: str) -> None:
    print(f"== {rel(folder)} ({aid}) ==")
    resolved = resolve_html(aid)
    if not resolved:
        print("  FAILED: no HTML source")
        return
    base, html = resolved
    figs = parse_figures(html)
    if not figs:
        print("  FAILED: parsed 0 figures")
        return
    print(f"  parsed figures: {sorted(figs)}")

    out_dir = folder / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)
    backup = out_dir / "_from_pdf_crop"
    # move existing fig*.png that aren't from html yet
    existing = list(out_dir.glob("fig*.png"))
    if existing and not (out_dir / ".from_html").exists():
        backup.mkdir(exist_ok=True)
        for p in existing:
            dest = backup / p.name
            if not dest.exists():
                shutil.move(str(p), str(dest))

    ok = 0
    for num in sorted(figs):
        srcs = figs[num]
        # prefer largest / last non-svg if multiple; try each until download works
        saved = False
        for src in srcs:
            url = abs_url(base, src)
            code, data = fetch(url)
            if code != 200 or len(data) < 200:
                print(f"  fig{num}: bad download {url} ({code}, {len(data)}B)")
                continue
            # skip tiny placeholders
            if len(data) < 500:
                continue
            ext = Path(src).suffix.lower() or ".png"
            if ext == ".svg":
                # keep svg as-is with png name? convert not available — save .svg and also note
                out = out_dir / f"fig{num:02d}.svg"
                out.write_bytes(data)
                # markdown expects png; try to keep as png if content is actually png
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    (out_dir / f"fig{num:02d}.png").write_bytes(data)
                    out.unlink(missing_ok=True)
                else:
                    print(f"  fig{num}: saved SVG only ({out.name}); md may need update")
                    saved = True
                    ok += 1
                    break
            out = out_dir / f"fig{num:02d}.png"
            # if jpeg, still write with .png only if PNG magic; else write real bytes with correct ext and copy
            if data[:8] == b"\x89PNG\r\n\x1a\n":
                out.write_bytes(data)
            elif data[:2] == b"\xff\xd8":
                jpg = out_dir / f"fig{num:02d}.jpg"
                jpg.write_bytes(data)
                # keep filename figXX.png for md compatibility via rewrite? use pillow if available
                try:
                    from PIL import Image
                    import io

                    im = Image.open(io.BytesIO(data))
                    im.save(out)
                except Exception:
                    shutil.copyfile(jpg, out.with_suffix(".jpg"))
                    print(f"  fig{num}: saved as jpg; update md if needed")
            else:
                out.write_bytes(data)
            print(f"  fig{num}: {out.name} <- {url} ({len(data)}B)")
            saved = True
            ok += 1
            break
        if not saved:
            print(f"  fig{num}: FAILED all srcs {srcs}")

    # Fill gaps from previous PDF crops if HTML missed some numbers
    backup = out_dir / "_from_pdf_crop"
    if backup.is_dir():
        for p in sorted(backup.glob("fig*.png")):
            dest = out_dir / p.name
            if not dest.exists():
                shutil.copy2(p, dest)
                print(f"  filled gap from pdf crop: {p.name}")

    (out_dir / ".from_html").write_text(base + "\n", encoding="utf-8")
    print(f"  done: {ok}/{len(figs)} figures from HTML; total png={len(list(out_dir.glob('fig*.png')))}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+", help="paper folder(s), as given or relative to docs/")
    ap.add_argument("--arxiv", help="arXiv id; only valid with a single folder")
    args = ap.parse_args(argv[1:])
    if args.arxiv and len(args.folders) != 1:
        ap.error("--arxiv needs exactly one folder")

    rc = 0
    for name in args.folders:
        folder = resolve(name)
        if folder is None or not folder.is_dir():
            print(f"  skip, not a directory: {name}")
            rc = 1
            continue
        aid = args.arxiv or arxiv_id(folder)
        if not aid:
            print(f"  skip {rel(folder)}: no *_<arxiv>.pdf found; pass --arxiv")
            rc = 1
            continue
        process(folder, aid)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
