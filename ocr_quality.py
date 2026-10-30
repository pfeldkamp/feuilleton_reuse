# Score the OCR quality of every installment (FR and DE) with Impresso's own method,
# so the re-OCR'd French installments get a score comparable to the others.
#
# Method (impresso-project/OCR-quality-assessment-unigram, v2.0.0):
#   score = share of distinct words in the text that are in a known-word list
#   (the word list is a Bloom filter built from Wikipedia + historical spellings)
#
# Check: on installments with Impresso OCR this reproduces the API's "ocr_quality"
#   DE: r = 1.00, mean difference 0.01   FR: r = 0.97, mean difference 0.04 (ours a bit higher)
#
# Output: adds "ocrqa" to every record in secret_du_titan/{fr,de}.jsonl
#         and prints mean scores per language and OCR source.
#
# Needs: pip install cython pybloomfiltermmap3 huggingface_hub
import json
import unicodedata
from pathlib import Path

from huggingface_hub import hf_hub_download
from pybloomfilter import BloomFilter

DATA = Path(__file__).parent / "secret_du_titan"
MODEL = "impresso-project/OCR-quality-assessment-unigram"
VERSION = "2.0.0"

# Normalisation from the model card: lowercase, punctuation -> space, digits -> 0
PUNCT = "„•<>!\"#%&'’()*,./:;?[]\\~_{}\xa1\xab\xb7\xbb\xbf—^`¦§£=-"
TABLE = str.maketrans({c: " " for c in PUNCT} | {d: "0" for d in "0123456789"})


def words(text):
    return set(unicodedata.normalize("NFKC", text).lower().translate(TABLE).split())


def ocrqa(text, known):
    w = words(text)
    return round(sum(x in known for x in w) / len(w), 3) if w else None


if __name__ == "__main__":  # so other scripts can import ocrqa() without rewriting the data
    for lang in ["fr", "de"]:
        known = BloomFilter.open(hf_hub_download(MODEL, f"ocrqa-wp_v{VERSION}-{lang}.bloom"))
        path = DATA / f"{lang}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        for rec in records:
            rec["ocrqa"] = ocrqa(rec["text"], known)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

        groups = {}
        for rec in records:
            groups.setdefault(rec.get("ocr_source", "impresso"), []).append(rec["ocrqa"])
        for source, scores in groups.items():
            print(f"{lang} {source:9s} n={len(scores):3d}  mean ocrqa={sum(scores) / len(scores):.3f}")
