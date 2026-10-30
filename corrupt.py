# %%
# Simulated OCR noise, roughly in the style of Impresso's historical French OCR:
# confused letters (e/c, n/u, rn/m, l/i/1, h/b ...), lost accents, dropped letters,
# stray punctuation, and words split or glued together.
#
# corrupt(text, rate) changes about `rate` of the characters (rate 0.05 ~ 5% character errors).
# Run this file to write corrupted copies of the round-trip French (which is clean,
# because Google fixed the OCR) to features/corrupted/, scored with Impresso's OCR-quality measure.
import json
import random
from pathlib import Path

CONFUSIONS = {  # letter -> things OCR often reads instead
    "e": ["c", "é", "o"], "c": ["e", "o"], "o": ["c", "e", "0"], "a": ["à", "o", "n"],
    "n": ["u", "m", "ri"], "u": ["n", "ii"], "m": ["rn", "in"], "i": ["l", "1", "î", "!"],
    "l": ["1", "i", "I"], "t": ["l", "f"], "f": ["t", "ſ"], "h": ["b", "li"], "b": ["h"],
    "s": ["a", "ſ"], "r": ["t", "i"], "d": ["cl"], "v": ["y"], "q": ["g"],
    "é": ["e", "è", "ê"], "è": ["é", "e"], "à": ["a"], "ê": ["e"], "ç": ["c"],
}
STRAY = list(".,'-;:^")
OPERATIONS = [("confuse", 0.55), ("drop", 0.15), ("stray", 0.10), ("space", 0.20)]


def corrupt(text, rate, seed=0):
    rng = random.Random(seed)
    ops, weights = zip(*OPERATIONS)
    out = []
    for ch in text:
        if rng.random() >= rate:
            out.append(ch)
            continue
        op = rng.choices(ops, weights)[0]
        low = ch.lower()
        if op == "confuse" and low in CONFUSIONS:
            new = rng.choice(CONFUSIONS[low])
            out.append(new.upper() if ch.isupper() else new)
        elif op == "drop":
            continue
        elif op == "stray":
            out.append(ch + rng.choice(STRAY))
        elif op == "space":
            out.append("" if ch == " " else ch + " ")  # glue two words, or split one
        else:
            out.append(ch)  # no confusion known for this character
    return "".join(out)


# %%
if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    from pybloomfilter import BloomFilter

    from ocr_quality import ocrqa  # Impresso's word-list OCR quality score

    CWD = Path(__file__).parent
    features = CWD / "secret_du_titan" / "features"
    out = features / "corrupted"
    out.mkdir(exist_ok=True)
    # Impresso's real French OCR of this novel scores 0.60-0.96 (median 0.885),
    # which is about 0-10% here; 15-30% shows what happens beyond that
    RATES = [0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]

    known = BloomFilter.open(hf_hub_download("impresso-project/OCR-quality-assessment-unigram",
                                             "ocrqa-wp_v2.0.0-fr.bloom"))
    source = [json.loads(l) for l in (features / "translations" / "fr_en_fr.jsonl").read_text().splitlines()]
    summary = {}
    for rate in RATES:
        recs = [{**{k: r[k] for k in ("id", "date", "ocr_source")},
                 "text": corrupt(r["text"], rate, seed=i)} for i, r in enumerate(source)]
        for r in recs:
            r["ocrqa"] = ocrqa(r["text"], known)
        name = f"fr_en_fr_noise{int(rate * 100):02d}"
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        summary[name] = {"rate": rate, "mean_ocrqa": round(sum(r["ocrqa"] for r in recs) / len(recs), 3)}
        print(name, summary[name])
    (out / "noise_levels.json").write_text(json.dumps(summary, indent=1))
