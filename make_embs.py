# %%
# Features for "Le Secret du Titan" (FR) and "Das Geheimniß" (DE):
#   1. translations of the French: FR->DE and a round trip FR->EN->FR (Google Translate)
#   2. embeddings (two Impresso GTE models)
#   3. named entities (Impresso NER)
#   4. 100 most frequent words (German texts only: real DE and FR->DE)
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
transl_output = data / "translations"
emb_output = data / "embeddings"
ner_output = data / "ner"
mfw_output = data / "mfw"
for folder in [transl_output, emb_output, ner_output, mfw_output]:
    folder.mkdir(exist_ok=True)

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
# 1. Translations: FR->DE and FR->EN->FR
from googletrans import Translator

translator = Translator()
MAX_CHARS = 4500  # Google's free endpoint refuses requests over ~5000 characters


def pieces(text, max_chars=MAX_CHARS):
    """Split a text into pieces under max_chars, at sentence ends."""
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?…])\s+", text):
        if cur and len(cur) + len(sent) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    return out + [cur] if cur else out


def translate(text, src, dest, tries=5):
    result = []
    for piece in pieces(text):
        for i in range(tries):
            try:
                result.append(translator.translate(piece, src=src, dest=dest).text)
                break
            except Exception as e:
                if i == tries - 1:
                    raise
                print(f"  retry {i + 1} after {type(e).__name__}", flush=True)
                time.sleep(10 * (i + 1))
        time.sleep(1)  # be polite to the free endpoint
    return " ".join(result)


def translate_fr_to_de(text):
    return translate(text, "fr", "de")


def translate_fr_to_en_to_fr(text):
    return translate(translate(text, "fr", "en"), "en", "fr")


for path, fn in [(tr_files[0], translate_fr_to_de), (tr_files[1], translate_fr_to_en_to_fr)]:
    done = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    for n, rec in enumerate(read_jsonl(fr_file), 1):
        if rec["id"] in done:
            continue
        done[rec["id"]] = {**meta(rec), "text": fn(rec["text"])}
        with path.open("a") as f:  # save as we go, so a crash loses nothing
            f.write(json.dumps(done[rec["id"]], ensure_ascii=False) + "\n")
        print(f"{path.name} {n} {rec['date']}", flush=True)
    print(f"{path.name}: {len(done)} installments")


# %%
# 2. Embeddings: one vector per installment
from sentence_transformers import SentenceTransformer

models = ['impresso-project/histlux-gte-multilingual-base',
          'impresso-project/OCR-diversely-robust-gte-multilingual-base']

for name in models:
    model = None
    for file in [de_file, fr_file] + tr_files:
        out = emb_output / f"{file.stem}__{name.split('/')[-1]}.jsonl"
        if out.exists() or not file.exists():
            continue
        model = model or SentenceTransformer(name, trust_remote_code=True)
        records = read_jsonl(file)
        embeddings = model.encode([r["text"] for r in records], show_progress_bar=True)
        write_jsonl(out, [{**meta(r), "embedding": e.tolist()} for r, e in zip(records, embeddings)])
        print(f"Saved embeddings for {file.name} using model {name}")


# %%
# 3. Named entities, with the same model for every text
from transformers import pipeline

ner = pipeline("impresso-ner", model="impresso-project/ner-hipe2020-mmbert-base-final",
               revision="impresso-ner-v1", trust_remote_code=True)


def get_NER(text):
    return ner(text, split_sentences=True)


for file in [de_file, fr_file] + tr_files:
    out = ner_output / f"{file.stem}.jsonl"
    if out.exists() or not file.exists():
        continue
    records = read_jsonl(file)
    write_jsonl(out, [{**meta(r), "entities": get_NER(r["text"])} for r in records])
    print(f"Saved NER for {file.name}")


# %%
# 4. 100 most frequent words, for the German texts (real DE vs FR->DE)
# MFW only compares texts in the same language, so the French is left out.
N_MFW = 100


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower())


german = {de_file.stem: read_jsonl(de_file), tr_files[0].stem: read_jsonl(tr_files[0])}
counts = {stem: [Counter(words(r["text"])) for r in recs] for stem, recs in german.items()}

# the word list comes from both German texts together, so both get the same 100 words
total = sum((c for cs in counts.values() for c in cs), Counter())
mfw = [w for w, _ in total.most_common(N_MFW)]

# relative frequency per installment, then z-score each word over all installments
freqs = {stem: np.array([[c[w] / max(sum(c.values()), 1) for w in mfw] for c in cs])
         for stem, cs in counts.items()}
stacked = np.vstack(list(freqs.values()))
mean, std = stacked.mean(axis=0), stacked.std(axis=0)

(mfw_output / "mfw_words.json").write_text(json.dumps(mfw, ensure_ascii=False))
for stem, recs in german.items():
    z = (freqs[stem] - mean) / std
    write_jsonl(mfw_output / f"{stem}.jsonl", [{**meta(r), "mfw_z": v.tolist()} for r, v in zip(recs, z)])
    print(f"Saved {N_MFW} MFW z-scores for {stem}")
print("top 20 MFW:", mfw[:20])

# %%
