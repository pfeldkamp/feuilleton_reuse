# %%
# Features for "Le Secret du Titan" (FR) and "Das Geheimniß" (DE):
#   1. translations of the French: FR->DE and a round trip FR->EN->FR (Google, via Sheets)
#   2. embeddings (two Impresso GTE models)
#   3. named entities (Impresso NER)
#   4. 100 most frequent words, within each language (DE vs FR->DE, FR vs FR->EN->FR)
#   5. passim-style text reuse score, within each language
#
# Every output is JSONL with one record per installment, keeping its id / date / ocr_source,
# so features can always be matched back to the text.
# Run cell by cell; each cell skips work that is already saved.

import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CWD = Path(__file__).parent
data = CWD / "secret_du_titan"
features = data / "features"  # everything this script makes goes in here
transl_output = features / "translations"
emb_output = features / "embeddings"
ner_output = features / "ner"
mfw_output = features / "mfw"
for folder in [transl_output, emb_output, ner_output, mfw_output]:
    folder.mkdir(parents=True, exist_ok=True)

de_file = data / "de.jsonl"
fr_file = data / "fr.jsonl"
tr_files = [transl_output / "fr_de.jsonl", transl_output / "fr_en_fr.jsonl"]

KEEP = ["id", "date", "ocr_source"]  # fields copied from the text into every feature file


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, records):
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False, default=float) + "\n" for r in records))


def meta(rec):
    return {k: rec.get(k) for k in KEEP}


# %%
# 1. Translations with Google Translate, through Google Sheets
# (the free Translate endpoint is CAPTCHA-blocked from this network;
#  Sheets' GOOGLETRANSLATE() uses the same engine)
#
# 1a writes sheets_in.csv: each installment cut into pieces, with the formulas already in it.
#    In Google Sheets: File > Import > Upload sheets_in.csv (keep "convert text to formulas" on),
#    wait until no cell says "Loading...", then File > Download > CSV,
#    and save it as features/translations/sheets_out.csv
# 1b reads sheets_out.csv and puts the pieces back together per installment.
import csv

MAX_CHARS = 2000  # per cell; GOOGLETRANSLATE gets unreliable on very long cells
sheets_in = transl_output / "sheets_in.csv"
sheets_out = transl_output / "sheets_out.csv"
fr_en = transl_output / "fr_en.jsonl"


def pieces(text, max_chars=MAX_CHARS):
    """Split a text into pieces under max_chars, at sentence ends."""
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?…])\s+", text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    return out + [cur] if cur else out


def sheet_safe(text):
    """One line, and no leading = + - @, which Sheets would read as a formula."""
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[=+@-]+\s*", "— ", text)


# %%
# 1a. Export for Google Sheets
rows = [["id", "piece", "fr", "de", "en", "en_fr"]]
for rec in read_jsonl(fr_file):
    for k, piece in enumerate(pieces(rec["text"])):
        r = len(rows) + 1  # the row number this line gets in the sheet
        rows.append([rec["id"], k, sheet_safe(piece),
                     f'=GOOGLETRANSLATE(C{r}, "fr", "de")',
                     f'=GOOGLETRANSLATE(C{r}, "fr", "en")',
                     f'=GOOGLETRANSLATE(E{r}, "en", "fr")'])
with sheets_in.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"wrote {sheets_in} with {len(rows) - 1} pieces")


# %%
# 1b. Import the translated sheet
if not sheets_out.exists():
    raise SystemExit(f"download the translated sheet as CSV to {sheets_out} first (see 1a)")

FAILED = ("#VALUE!", "#N/A", "#ERROR!", "Loading...", "")
with sheets_out.open(newline="") as f:
    sheet = list(csv.DictReader(f))
bad = [(row["id"], row["piece"], col) for row in sheet for col in ["de", "en", "en_fr"]
       if row[col].strip() in FAILED or row[col].startswith("#")]
if bad:
    print(f"{len(bad)} cells did not translate, e.g. {bad[:5]}; fix them in Sheets and download again")

by_id = {}
for row in sheet:
    by_id.setdefault(row["id"], []).append(row)
for out_file, col in [(tr_files[0], "de"), (fr_en, "en"), (tr_files[1], "en_fr")]:
    write_jsonl(out_file, [{**meta(rec), "text": " ".join(
        row[col] for row in sorted(by_id.get(rec["id"], []), key=lambda r: int(r["piece"])))}
        for rec in read_jsonl(fr_file)])
    print(f"saved {out_file.name}")


# %%
# 2. Embeddings: one vector per installment
# Also embeds the English step and the corrupted round trips (made by corrupt.py).
# One model at a time, small batches, to keep memory low (~1 s per installment on the M3).
import torch
from sentence_transformers import SentenceTransformer

models = ['impresso-project/histlux-gte-multilingual-base']
# not run, for lack of time (~1.5 h): 'impresso-project/OCR-diversely-robust-gte-multilingual-base'
device = "mps" if torch.backends.mps.is_available() else "cpu"
to_embed = ([de_file, fr_file] + tr_files + [transl_output / "fr_en.jsonl"]
            + sorted((features / "corrupted").glob("fr_en_fr_noise*.jsonl")))

for name in models:
    model = None
    for file in to_embed:
        out = emb_output / f"{file.stem}__{name.split('/')[-1]}.jsonl"
        if out.exists() or not file.exists():
            continue
        model = model or SentenceTransformer(name, trust_remote_code=True, device=device)
        records = read_jsonl(file)
        embeddings = model.encode([r["text"] for r in records], batch_size=2, show_progress_bar=False)
        write_jsonl(out, [{**meta(r), "embedding": e.tolist()} for r, e in zip(records, embeddings)])
        print(f"Saved embeddings for {file.name} using model {name}", flush=True)
    del model  # free it before the next model is loaded
    if device == "mps":
        torch.mps.empty_cache()


# %%
# 3. Named entities, with the same model for every text
# impresso-project/ner-hipe2020-mmbert-base-final needs transformers >= 5, which would break
# sentence-transformers 4.1 here; so we use Impresso's earlier multilingual NER model.
# It only reads 512 tokens at a time, so each installment goes in as ~1000-character pieces.
# Names are kept as written (no Pierre <-> Peter mapping): the realistic case.
from transformers import AutoTokenizer, pipeline

NER_MODEL = "impresso-project/ner-stacked-bert-multilingual"
ner = pipeline("generic-ner", model=NER_MODEL, tokenizer=AutoTokenizer.from_pretrained(NER_MODEL),
               trust_remote_code=True, device="cpu")


def text_pieces(text, max_chars=1000):
    """Pieces under max_chars, cut at sentence ends, with their start offset in the text."""
    out, start, cur = [], 0, ""
    for m in re.finditer(r"[^.!?…]+[.!?…]*\s*", text):
        if cur and len(cur) + len(m.group()) > max_chars:
            out.append((start, cur))
            start, cur = m.start(), ""
        cur += m.group()
    return out + [(start, cur)] if cur else out


def get_NER(text):
    entities = []
    for offset, piece in text_pieces(text):
        for e in ner(piece):
            entities.append({**e, "lOffset": e["lOffset"] + offset, "rOffset": e["rOffset"] + offset})
    return entities


for file in [de_file, fr_file]:  # the two originals; the translations are not needed for DE <-> FR
    out = ner_output / f"{file.stem}.jsonl"
    if out.exists() or not file.exists():
        continue
    records = read_jsonl(file)
    write_jsonl(out, [{**meta(r), "entities": get_NER(r["text"])} for r in records])
    print(f"Saved NER for {file.name}", flush=True)


# %%
# Within-language comparisons (cells 4 and 5): each pair is in one language,
# so word-level features (MFW, passim n-grams) can compare them directly.
# The feature code itself is in textfeatures.py (shared with compare_features.py).
from textfeatures import mfw_z, passim_matrix

SAME_LANGUAGE = [(de_file, tr_files[0]),   # real German vs Google's German
                 (fr_file, tr_files[1])]   # French original vs FR->EN->FR round trip


# %%
# 4. 100 most frequent words (MFW), per language pair
N_MFW = 100
mfw_lists = {}

for a, b in SAME_LANGUAGE:
    if not b.exists():
        print(f"skipping MFW for {a.name}: {b.name} not made yet (cell 1)")
        continue
    recs = {a.stem: read_jsonl(a), b.stem: read_jsonl(b)}
    z_a, z_b, mfw = mfw_z([r["text"] for r in recs[a.stem]], [r["text"] for r in recs[b.stem]], N_MFW)
    mfw_lists[f"{a.stem}__{b.stem}"] = mfw
    for stem, z in [(a.stem, z_a), (b.stem, z_b)]:
        write_jsonl(mfw_output / f"{a.stem}__{b.stem}__{stem}.jsonl",
                    [{**meta(r), "mfw_z": v.tolist()} for r, v in zip(recs[stem], z)])
    print(f"Saved {N_MFW} MFW z-scores for {a.stem} vs {b.stem}; top 10: {mfw[:10]}")

(mfw_output / "mfw_words.json").write_text(json.dumps(mfw_lists, ensure_ascii=False, indent=1))


# %%
# 5. passim-style text reuse score, per language pair
# Score of installment x against y = share of x's rare 5-grams that also occur in y.
N_GRAM = 5    # passim's default n-gram length
MAX_DF = 5    # drop n-grams found in more than 5 installments (headers, set phrases)

passim_output = features / "passim"
passim_output.mkdir(exist_ok=True)

for a, b in SAME_LANGUAGE:
    if not b.exists():
        print(f"skipping passim for {a.name}: {b.name} not made yet (cell 1)")
        continue
    recs_a, recs_b = read_jsonl(a), read_jsonl(b)
    sim = passim_matrix([r["text"] for r in recs_a], [r["text"] for r in recs_b], N_GRAM, MAX_DF)
    out = [{**meta(r), "scores": {rb["id"]: round(float(v), 4) for rb, v in zip(recs_b, row) if v > 0}}
           for r, row in zip(recs_a, sim)]
    write_jsonl(passim_output / f"{a.stem}__{b.stem}.jsonl", out)
    print(f"Saved passim-style scores for {a.stem} vs {b.stem} "
          f"({sum(bool(o['scores']) for o in out)}/{len(out)} installments share any rare {N_GRAM}-gram)")

# %%
