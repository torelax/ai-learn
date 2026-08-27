#!/usr/bin/env python3
"""Reorganize docs/papers into per-paper / per-series folders (no arXiv IDs).

Layout:
  docs/papers/<topic>/<Name>/{paper.pdf, zh.md, figs/}
  docs/papers/<topic>/<Series>/<Paper>/{paper.pdf, zh.md, figs/}

Scripts (_fetch_html_figures.py, _embed_figures.py) stay at topic root.
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path("/data/zhangchangtian/project/ai-learn/docs/papers")

# topic, dest relative to topic, old stem (files: STEM.pdf, STEM_zh.md, STEM_figs/)
SERIES: list[tuple[str, str, str]] = [
    ("embedding", "GTE/GTE", "GTE_2308.03281"),
    ("embedding", "BGE/C-Pack", "BGE-CPack_2309.07597"),
    ("embedding", "BGE/M3", "BGE-M3_2402.03216"),
    ("embedding", "BGE/EN-ICL", "BGE-EN-ICL_2409.15700"),
    ("embedding", "Conan-embedding/v1", "Conan-embedding-v1_2408.15710"),
    ("embedding", "Conan-embedding/v2", "Conan-embedding-v2_2509.12892"),
    ("embedding", "Arctic-Embed/v1", "Arctic-Embed_2405.05374"),
    ("embedding", "Arctic-Embed/v2", "Arctic-Embed-v2_2412.04506"),
    ("embedding", "DINO/DINOv2", "DINOv2_2304.07193"),
    ("embedding", "DINO/DINOv3", "DINOv3_2508.10104"),
    ("embedding", "RocketQA/RocketQA", "RocketQA_2010.08191"),
    ("embedding", "RocketQA/RocketQAv2", "RocketQAv2_2110.07367"),
    ("embedding", "SigLIP/SigLIP", "SigLIP_2303.15343"),
    ("embedding", "SigLIP/SigLIP2", "SigLIP2_2502.14786"),
    ("embedding", "Condenser/Condenser", "Condenser_2104.08253"),
    ("embedding", "Condenser/coCondenser", "coCondenser_2108.05540"),
]

STANDALONE: list[tuple[str, str, str]] = [
    ("embedding", "ANCE", "ANCE_2007.00808"),
    ("embedding", "CLIP", "CLIP_2103.00020"),
    ("embedding", "Contriever", "Contriever_2112.09118"),
    ("embedding", "DeVE-QA", "DeVE-QA_2409.04388"),
    ("embedding", "DupMAE", "DupMAE_2211.08769"),
    ("embedding", "E5-Mistral", "E5-Mistral_2401.00368"),
    ("embedding", "GME", "GME_2412.16855"),
    ("embedding", "GritLM", "GritLM_2402.13342"),
    ("embedding", "INSTRUCTOR", "INSTRUCTOR_2212.09741"),
    ("embedding", "InternLM2", "InternLM2_2403.17297"),
    ("embedding", "LateChunking", "LateChunking_2409.04701"),
    ("embedding", "LLM2Vec", "LLM2Vec_2404.05961"),
    ("embedding", "LLM-DA-TPR", "LLM-DA-TPR_2405.11971"),
    ("embedding", "MegaPairs", "MegaPairs_2412.14475"),
    ("embedding", "ModernBERT", "ModernBERT_2412.13663"),
    ("embedding", "NV-Embed", "NV-Embed_2405.17428"),
    ("embedding", "NV-Retriever", "NV-Retriever_2407.15831"),
    ("embedding", "Qwen3-Embedding", "Qwen3-Embedding_2506.05176"),
    ("embedding", "QZhou-Embedding", "QZhou-Embedding_2508.21632"),
    ("embedding", "RetroMAE", "RetroMAE_2205.12035"),
    ("embedding", "SimCSE", "SimCSE_2104.08821"),
    ("embedding", "Token-Prepending", "Token-Prepending_ACL2025"),
    ("embedding", "Vec2Vec", "Vec2Vec_2505.12540"),
    ("embedding", "ViDoRev2", "ViDoRev2_2505.17166"),
    ("embedding", "VLM2Vec", "VLM2Vec_2410.05160"),
    ("contrastive", "LilianWeng-contrastive", "LilianWeng_contrastive_2021"),
]


def move_into(topic: str, dest_rel: str, stem: str) -> None:
    src_dir = ROOT / topic
    dest = ROOT / topic / dest_rel
    dest.mkdir(parents=True, exist_ok=True)
    moved = []

    pdf = src_dir / f"{stem}.pdf"
    if pdf.is_file():
        target = dest / "paper.pdf"
        if not target.exists():
            shutil.move(str(pdf), str(target))
            moved.append("paper.pdf")

    zh = src_dir / f"{stem}_zh.md"
    if zh.is_file():
        target = dest / "zh.md"
        if not target.exists():
            text = zh.read_text(encoding="utf-8")
            text = text.replace(f"{stem}_figs/", "figs/")
            target.write_text(text, encoding="utf-8")
            zh.unlink()
            moved.append("zh.md")

    figs = src_dir / f"{stem}_figs"
    if figs.is_dir():
        target = dest / "figs"
        if target.exists():
            for p in figs.iterdir():
                d = target / p.name
                if not d.exists():
                    shutil.move(str(p), str(d))
            shutil.rmtree(figs, ignore_errors=True)
        else:
            shutil.move(str(figs), str(target))
        moved.append("figs/")

    print(f"  {stem} -> {topic}/{dest_rel}  [{', '.join(moved) or 'nothing'}]")


def main() -> int:
    print("== series ==")
    for topic, dest, stem in SERIES:
        move_into(topic, dest, stem)
    print("== standalone ==")
    for topic, dest, stem in STANDALONE:
        move_into(topic, dest, stem)

    # leftover flat files?
    for topic in ("embedding", "contrastive", "lora", "Qwen", "distillation"):
        d = ROOT / topic
        if not d.is_dir():
            continue
        leftovers = []
        for p in sorted(d.iterdir()):
            if p.name.startswith("_") or p.name.startswith("."):
                continue
            if p.suffix in {".pdf", ".md"} or p.name.endswith("_figs"):
                leftovers.append(p.name)
            if p.is_dir() and list(p.glob("*.pdf")):
                pass
        if leftovers:
            print(f"LEFTOVER in {topic}: {leftovers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
