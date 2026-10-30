#!/usr/bin/env python3
"""Compare Tesseract (text.txt) and PERO (pero.txt) with the Impresso OCR-quality metric.

Same metric as the Impresso script: share of distinct normalised word types found in the
Wikipedia-derived Bloom filter (impresso-project/OCR-quality-assessment-unigram v2.0.0).

    pip install huggingface_hub pybloomfiltermmap3
    python eval_ocrqa.py               # writes ocrqa_scores.csv + prints summary
    python eval_ocrqa.py --dehyphen    # also rejoin "mo-\\nde" (see caveat below)

Default is NO dehyphenation: these text.txt files have lost almost all line-end hyphens
("qu'au\\ncun"), so rejoining only where a hyphen survives would favour PERO, which keeps
them. Without it, "-" becomes a space and both engines are penalised for the same fragments.
"""
import argparse, csv, re, statistics, unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download
from pybloomfilter import BloomFilter

ROOT = Path(__file__).parent
MODEL = "impresso-project/OCR-quality-assessment-unigram"
VERSION = "2.0.0"
LANG = "fr"
SOURCES = {"tesseract": "text.txt", "pero": "pero.txt"}

# Normalisation from the model card: lowercase, punctuation -> space, digits -> 0
PUNCT = "„•<>!\"#%&'’()*,./:;?[]\\~_{}\xa1\xab\xb7\xbb\xbf—^`¦§£=-"
TABLE = str.maketrans({c: " " for c in PUNCT} | {d: "0" for d in "0123456789"})


def dehyphen(text):
    # rejoin words split across lines: "inter-\nrogatoire" -> "interrogatoire"
    return re.sub(r"(\w)[-¬]\s*\n\s*(\w)", r"\1\2", text)


def words(text):
    return set(unicodedata.normalize("NFKC", text).lower().translate(TABLE).split())


def ocrqa(text, known):
    w = words(text)
    return (round(sum(x in known for x in w) / len(w), 3) if w else None), len(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dehyphen", action="store_true")
    a = ap.parse_args()

    known = BloomFilter.open(hf_hub_download(MODEL, f"ocrqa-wp_v{VERSION}-{LANG}.bloom"), "r")
    rows = []
    for folder in sorted(ROOT.glob("depechetoulouse-*")):
        row = {"issue": folder.name}
        for src, fname in SOURCES.items():
            p = folder / fname
            if not p.exists():
                row[f"{src}_ocrqa"] = row[f"{src}_types"] = row[f"{src}_tokens"] = None
                continue
            text = p.read_text(encoding="utf-8")
            if a.dehyphen:
                text = dehyphen(text)
            score, ntypes = ocrqa(text, known)
            row[f"{src}_ocrqa"] = score
            row[f"{src}_types"] = ntypes
            row[f"{src}_tokens"] = len(unicodedata.normalize("NFKC", text).lower().translate(TABLE).split())
        if row.get("tesseract_ocrqa") is not None and row.get("pero_ocrqa") is not None:
            row["delta_pero_minus_tess"] = round(row["pero_ocrqa"] - row["tesseract_ocrqa"], 3)
        rows.append(row)

    out = ROOT / "ocrqa_scores.csv"
    fields = ["issue"] + [f"{s}_{k}" for s in SOURCES for k in ("ocrqa", "types", "tokens")] + ["delta_pero_minus_tess"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['issue']}  tess={r['tesseract_ocrqa']}  pero={r['pero_ocrqa']}  "
              f"tokens tess/pero={r['tesseract_tokens']}/{r['pero_tokens']}")
    for src in SOURCES:
        s = [r[f"{src}_ocrqa"] for r in rows if r[f"{src}_ocrqa"] is not None]
        if s:
            print(f"{LANG} {src:9s} n={len(s):3d}  mean ocrqa={statistics.mean(s):.3f}  median={statistics.median(s):.3f}")
    d = [r["delta_pero_minus_tess"] for r in rows if r.get("delta_pero_minus_tess") is not None]
    if d:
        print(f"paired delta (pero - tesseract): mean={statistics.mean(d):+.3f}, "
              f"PERO better on {sum(x > 0 for x in d)}/{len(d)} issues")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
