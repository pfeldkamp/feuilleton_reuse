# %%
# How well does each feature find the matching installment?
#
# Main measure, "progression": Spearman correlation between where each installment really is
# in the novel and where its best match is. 1 = every installment found in the right order.
# Also top-1: share of installments whose best match is the right one.
#
# 1. comparison.json - the text pairs:
#    fr vs fr_en_fr  French vs its round trip (FR->EN->FR)     right answer known (same id)
#    fr vs fr_en     French vs its English translation          right answer known (same id)
#    de vs fr_de     real German (1891) vs Google's German        right answer approximate:
#    de vs fr        real German vs the French directly           the installment at the same
#                                                                 relative position, +-1
# 2. robustness.json - French vs the round trip with simulated OCR noise (corrupt.py),
#    from 0% to 30% of characters changed.
#
# MFW and passim only compare texts in the same language; embeddings compare any pair.
# NER (names as written, TF-IDF) is only extracted for the two originals, fr and de.
import json
from pathlib import Path

import numpy as np

from textfeatures import align_in_order, combine, cosine_matrix, mfw_z, name_bag, passim_matrix, score, tfidf_cosine

CWD = Path(__file__).parent
data = CWD / "secret_du_titan"
features = data / "features"
MODELS = ["histlux-gte-multilingual-base"]  # the OCR-robust model was not run (time)
SAME_LANGUAGE = {("fr", "fr_en_fr"), ("de", "fr_de")} | {("fr", p.stem) for p in (features / "corrupted").glob("*.jsonl")}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def find(stem):
    for folder in [data, features / "translations", features / "corrupted"]:
        if (folder / f"{stem}.jsonl").exists():
            return folder / f"{stem}.jsonl"
    raise FileNotFoundError(stem)


def texts(stem):
    return [r["text"] for r in read_jsonl(find(stem))]


def embedding_matrix(a, b, model):
    fa, fb = features / "embeddings" / f"{a}__{model}.jsonl", features / "embeddings" / f"{b}__{model}.jsonl"
    if not (fa.exists() and fb.exists()):
        return None
    return cosine_matrix([r["embedding"] for r in read_jsonl(fa)], [r["embedding"] for r in read_jsonl(fb)])


def similarity_matrices(a, b):
    """{feature name: similarity matrix (rows = a's installments, columns = b's)}"""
    sims = {}
    if (a, b) in SAME_LANGUAGE:
        z_a, z_b, _ = mfw_z(texts(a), texts(b))
        sims["MFW"] = cosine_matrix(z_a, z_b)
        sims["passim"] = passim_matrix(texts(a), texts(b))
    for model in MODELS:
        sim = embedding_matrix(a, b, model)
        if sim is not None:
            sims["GTE " + ("histlux" if model.startswith("histlux") else "OCR-robust")] = sim
    ner_a, ner_b = features / "ner" / f"{a}.jsonl", features / "ner" / f"{b}.jsonl"
    if ner_a.exists() and ner_b.exists():
        ents_a = [r["entities"] for r in read_jsonl(ner_a)]
        ents_b = [r["entities"] for r in read_jsonl(ner_b)]
        for unit in ["full names", "name words"]:
            sims[f"NER ({unit})"] = tfidf_cosine([name_bag(e, unit) for e in ents_a],
                                                 [name_bag(e, unit) for e in ents_b])
    if "GTE histlux" in sims and "NER (name words)" in sims:
        sims["GTE + NER"] = combine(sims["GTE histlux"], sims["NER (name words)"])
    return sims


def positions(stem):
    """Middle of each installment as a share of the whole novel (0-1), by characters."""
    lengths = np.array([len(t) for t in texts(stem)])
    ends = np.cumsum(lengths)
    return (ends - lengths / 2) / ends[-1]


def expected_match(a, b):
    """For each installment of a, the index of b's installment at the same relative position."""
    pos_b = positions(b)
    return np.array([np.abs(pos_b - p).argmin() for p in positions(a)])


# %%
# 1. The text pairs
PAIRS = [("fr", "fr_en_fr", "known"), ("fr", "fr_en", "known"),
         ("de", "fr_de", "position"), ("de", "fr", "position")]

comparison = []
for a, b, answer in PAIRS:
    expected = np.arange(len(texts(a))) if answer == "known" else expected_match(a, b)
    tolerance = 0 if answer == "known" else 1
    for feature, sim in similarity_matrices(a, b).items():
        comparison.append({"pair": f"{a} vs {b}", "answer": answer, "feature": feature,
                           **score(sim, expected, tolerance),
                           # kept for plotting the mapping: each installment's expected and best match
                           "expected": expected.tolist(), "best": sim.argmax(axis=1).tolist(),
                           "n_candidates": sim.shape[1]})
        if answer == "position":  # the same similarities, with the matches forced to stay in order
            path = align_in_order(sim)
            comparison.append({"pair": f"{a} vs {b}", "answer": answer, "feature": f"{feature} (in order)",
                               **score(sim, expected, tolerance, best=path),
                               "expected": expected.tolist(), "best": path, "n_candidates": sim.shape[1]})

print(f"{'pair':16s} {'feature':28s} {'progression':>11s} {'top-1':>6s} {'chance':>7s} {'med.error':>10s}")
for r in comparison:
    print(f"{r['pair']:16s} {r['feature']:28s} {r['progression']:11.2f} {r['top1']:6.2f} "
          f"{r['chance_top1']:7.3f} {r['median_error']:10.1f}")
(features / "comparison.json").write_text(json.dumps(comparison, indent=1))


# %%
# 2. Robustness to OCR noise: French vs the corrupted round trips
levels = json.loads((features / "corrupted" / "noise_levels.json").read_text())
robustness = []
for stem, info in sorted(levels.items(), key=lambda kv: kv[1]["rate"]):
    expected = np.arange(len(texts("fr")))
    for feature, sim in similarity_matrices("fr", stem).items():
        robustness.append({"noise": info["rate"], "ocrqa": info["mean_ocrqa"], "feature": feature,
                           **score(sim, expected)})

names = list(dict.fromkeys(r["feature"] for r in robustness))
for measure in ["progression", "separation"]:
    print(f"\n{measure} by noise level")
    print(f"{'noise':>6s} {'ocrqa':>6s}  " + "  ".join(f"{f:>16s}" for f in names))
    for rate in sorted({r["noise"] for r in robustness}):
        row = {r["feature"]: r[measure] for r in robustness if r["noise"] == rate}
        q = next(r["ocrqa"] for r in robustness if r["noise"] == rate)
        print(f"{rate:6.3f} {q:6.3f}  " + "  ".join(f"{row.get(f, float('nan')):16.2f}" for f in names))
(features / "robustness.json").write_text(json.dumps(robustness, indent=1))


# %%
# 3. The German -> French mapping, one row per German installment
import csv

de, fr = read_jsonl(find("de")), read_jsonl(find("fr"))
rows = {r["feature"]: r for r in comparison if r["pair"] == "de vs fr"}
with (features / "mapping_de_fr.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["de_id", "de_date", "fr_gte_ner_in_order", "fr_gte_ner_date", "fr_gte_in_order", "fr_gte_date",
                "fr_by_position", "fr_by_position_date", "fr_ocr_source"])
    for i, rec in enumerate(de):
        a = fr[rows["GTE + NER (in order)"]["best"][i]]
        b = fr[rows["GTE histlux (in order)"]["best"][i]]
        c = fr[rows["GTE histlux"]["expected"][i]]
        w.writerow([rec["id"], rec["date"], a["id"], a["date"], b["id"], b["date"], c["id"], c["date"], a["ocr_source"]])
print("saved mapping_de_fr.csv")
