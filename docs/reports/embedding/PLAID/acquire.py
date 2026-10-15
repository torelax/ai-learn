#!/usr/bin/env python3
"""Fetch PLAID sources, preserving all panels and recording actual outcomes.

Run with /workspace/venvs/ai-learn/bin/python (PyMuPDF, Pillow, bs4).
No translation is generated and no human review is claimed by this script.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.parse import urljoin, urlparse

import fitz
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parent
PAPER_ID = "2205.09707"
PDF_NAME = f"PLAID: An Efficient Engine for Late Interaction Retrieval_{PAPER_ID}.pdf"
spec = importlib.util.spec_from_file_location("existing_fetch", ROOT.parent / "_fetch_html_figures.py")
existing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing)


def keep(path: Path, data: bytes) -> None:
    """Never silently overwrite an existing source or user file."""
    if path.exists() and path.read_bytes() != data:
        raise FileExistsError(f"Existing content differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    log = {"paper_id": PAPER_ID, "at_utc": datetime.now(timezone.utc).isoformat(),
           "requests": [], "pdf": None, "html": None, "images": [], "blockers": []}

    def fetch(url: str):
        code, data = existing.fetch(url, timeout=15)
        log["requests"].append({"url": url, "status": code, "bytes": len(data),
                                "success": code == 200})
        return code, data

    for url in [f"https://arxiv.org/pdf/{PAPER_ID}",
                f"https://export.arxiv.org/pdf/{PAPER_ID}",
                "https://cs.stanford.edu/~matei/papers/2022/cikm_plaid.pdf"]:
        code, data = fetch(url)
        if code != 200 or not data.startswith(b"%PDF-"):
            continue
        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                first = doc[0].get_text()
                if "PLAID" not in first or "Late Interaction" not in first:
                    raise ValueError("Downloaded PDF title does not match PLAID")
                text = "\n\n".join(f"<!-- PDF page {i+1} -->\n{p.get_text()}" for i, p in enumerate(doc))
                # Caption candidates require manual checking against page images.
                nums = sorted(set(int(m.group(1)) for m in re.finditer(
                    r"(?m)^\s*(?:Figure|Fig\.?)\s+(\d+)\s*[:.]", text)))
                pages = len(doc)
            keep(ROOT / PDF_NAME, data)
            keep(ROOT / "sources" / "pdf-text.en.txt", text.encode())
            log["pdf"] = {"url": url, "file": PDF_NAME, "sha256": sha(data),
                          "pages": pages, "caption_candidates": nums,
                          "version_alignment": "requires manual PDF/HTML verification"}
            break
        except Exception as exc:
            log["blockers"].append(f"PDF validation: {exc}")

    for url in [f"https://arxiv.org/html/{PAPER_ID}",
                f"https://ar5iv.labs.arxiv.org/html/{PAPER_ID}",
                f"https://ar5iv.org/html/{PAPER_ID}"]:
        code, data = fetch(url)
        if code != 200:
            continue
        html = data.decode("utf-8", errors="replace")
        if "PLAID" not in html or "ltx_figure" not in html:
            continue
        legacy_figures = existing.parse_figures(html)
        soup = BeautifulSoup(html, "html.parser")
        figures = {}
        panel_metadata = {}
        captions = {}
        all_numbers = set()
        for figure in soup.select("figure.ltx_figure"):
            number = re.fullmatch(r".*\.F(\d+)", figure.get("id", ""))
            if not number:  # Nested subfigures are included in their parent.
                continue
            n = int(number.group(1))
            all_numbers.add(n)
            caption = figure.find("figcaption", recursive=False)
            captions[str(n)] = caption.get_text(" ", strip=True) if caption else "MISSING CAPTION"
            figures[n] = []
            panel_metadata[n] = []
            for asset in figure.select("img[src], object[data]"):
                src = asset.get("src") or asset["data"]
                figures[n].append(src)
                parent = asset.find_parent("figure")
                subcaption = parent.find("figcaption", recursive=False) if parent else None
                panel_metadata[n].append({"html_id": asset.get("id"),
                                          "caption": subcaption.get_text(" ", strip=True) if subcaption else "",
                                          "legend": "legend" in src})
        keep(ROOT / "sources" / "paper.html", data)
        base_match = re.search(r'<base\b[^>]*href=[\"\']([^\"\']+)', html, re.I)
        base = urljoin(url, unescape(base_match.group(1))) if base_match else url
        log["html"] = {"url": url, "sha256": sha(data), "captions": captions,
                       "caption_numbers": sorted(all_numbers), "parser_numbers": sorted(figures),
                       "legacy_parser_numbers": sorted(legacy_figures), "panel_metadata": panel_metadata}
        if all_numbers != set(figures):
            log["blockers"].append("HTML caption/asset mapping mismatch; inspect inline SVG, tables or parser omissions")
        for num, srcs in sorted(figures.items()):
            for panel, src in enumerate(srcs, 1):
                asset_url = urljoin(base, unescape(src))
                code, content = fetch(asset_url)
                record = {"figure": num, "panel": panel, "url": asset_url, "status": "failed"}
                record.update(panel_metadata[num][panel - 1])
                log["images"].append(record)
                if code != 200:
                    continue
                try:
                    stem = f"fig{num:02d}-panel{panel:02d}"
                    is_svg = b"<svg" in content[:2000] or urlparse(asset_url).path.lower().endswith(".svg")
                    target = ROOT / "figs" / f"{stem}.png"
                    if is_svg:
                        original = ROOT / "figs" / f"{stem}.svg"
                        keep(original, content)
                        if not target.exists():
                            converter_env = dict(os.environ, INKSCAPE_PROFILE_DIR="/workspace/.config/inkscape",
                                                 XDG_CACHE_HOME="/workspace/.cache")
                            subprocess.run(["inkscape", str(original), "--export-type=png",
                                            "--export-dpi=192", f"--export-filename={target}"],
                                           env=converter_env, check=True, capture_output=True, timeout=60)
                    else:
                        im = Image.open(io.BytesIO(content))
                        original = ROOT / "figs" / f"{stem}.{im.format.lower()}"
                        keep(original, content)
                        if not target.exists():
                            im.convert("RGBA").save(target, format="PNG")
                    with Image.open(target) as im:
                        im.verify()
                    with Image.open(target) as im:
                        record.update(status="decoded", size=list(im.size), format=im.format,
                                      png=str(target.relative_to(ROOT)), png_sha256=sha(target.read_bytes()),
                                      original_sha256=sha(content), visual_review="pending")
                except Exception as exc:
                    record["error"] = str(exc)
        break

    if log["pdf"] is None:
        log["blockers"].append("PDF unavailable: no full-text translation, PDF figure inventory or original equation audit possible")
    if log["html"] is None:
        log["blockers"].append("HTML unavailable: no original HTML images or paper SVG conversion possible")
    if any(i["status"] != "decoded" for i in log["images"]):
        log["blockers"].append("At least one HTML figure panel failed")
    if log["pdf"] and log["html"]:
        if log["pdf"]["caption_candidates"] != log["html"]["caption_numbers"]:
            log["blockers"].append("PDF/HTML figure-number mismatch; manual inventory required")
    log["outcome"] = "blocked" if log["blockers"] else "assets_retrieved_manual_review_pending"
    logs = ROOT / "sources" / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    filename = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    (logs / filename).write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return 2 if log["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
