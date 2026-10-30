#!/usr/bin/env python3
"""Run PERO OCR on every depechetoulouse-*/r*.jpg and write pero.txt next to text.txt.

Usage:
    python run_pero.py --config /path/to/pero_eu_cz_print_newspapers_2022-09-26/config_cpu.ini
Options:
    --root DIR     folder holding the depechetoulouse-* subfolders (default: this script's folder)
    --device cpu|gpu   (default cpu; Apple Silicon has no CUDA, so keep cpu)
    --overwrite    redo folders that already have pero.txt
Also writes pero_alto/rXX.xml (ALTO) per image, handy for later alignment.
"""
import argparse, configparser, glob, os, sys, time

import cv2
import torch
from pero_ocr.core.layout import PageLayout
from pero_ocr.document_ocr.page_parser import PageParser


def page_text(layout: PageLayout) -> str:
    paras = []
    for region in layout.regions:
        lines = [l.transcription for l in region.lines if l.transcription]
        if lines:
            paras.append("\n".join(lines))
    return "\n\n".join(paras)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    cfg_path = os.path.abspath(a.config)
    config = configparser.ConfigParser()
    config.read(cfg_path)
    device = torch.device("cuda" if a.device == "gpu" else "cpu")
    parser = PageParser(config, config_path=os.path.dirname(cfg_path), device=device)

    folders = sorted(glob.glob(os.path.join(a.root, "depechetoulouse-*")))
    if not folders:
        sys.exit(f"No depechetoulouse-* folders under {a.root}")
    for folder in folders:
        out = os.path.join(folder, "pero.txt")
        if os.path.exists(out) and not a.overwrite:
            print(f"skip {os.path.basename(folder)} (pero.txt exists)")
            continue
        alto_dir = os.path.join(folder, "pero_alto")
        os.makedirs(alto_dir, exist_ok=True)
        chunks = []
        t0 = time.time()
        for img_path in sorted(glob.glob(os.path.join(folder, "r*.jpg"))):
            name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, 1)
            layout = PageLayout(id=name, page_size=(img.shape[0], img.shape[1]))
            layout = parser.process_page(img, layout)
            layout.to_altoxml(os.path.join(alto_dir, name + ".xml"))
            chunks.append(page_text(layout))
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n\n".join(c for c in chunks if c).strip() + "\n")
        print(f"{os.path.basename(folder)}: {len(chunks)} images, {time.time()-t0:.0f}s -> pero.txt")


if __name__ == "__main__":
    main()
