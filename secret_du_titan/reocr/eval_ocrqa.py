#!/usr/bin/env python3
"""Compare Tesseract (text.txt) and PERO (pero.txt) with the Impresso OCR-quality metric (OCRQA).

OCRQA = share of the distinct normalised words of a text that are found in a French lexicon
built from Wikipedia (Bloom filter impresso-project/OCR-quality-assessment-unigram v2.0.0).

    pip install huggingface_hub pybloomfiltermmap3 matplotlib
    python eval_ocrqa.py                # score every issue, then write the report + chart
    python eval_ocrqa.py --report-only  # rebuild report + chart from an existing ocrqa_scores.csv
    python eval_ocrqa.py --dehyphen     # also rejoin "mo-\\nde" (not recommended, see below)

Outputs (in this folder):
    ocrqa_scores.csv      one row per issue, both engines
    ocrqa_summary.csv     one row per engine (mean, median, ... ) - the headline table
    ocrqa_report.md       results + plain-language explanation of every column
    ocrqa_comparison.png  per-issue chart, Tesseract vs PERO

Default is NO dehyphenation: these text.txt files have lost almost all line-end hyphens
("qu'au\\ncun"), so rejoining only where a hyphen survives would favour PERO, which keeps
them. Without it, "-" becomes a space and both engines are penalised for the same fragments.
"""
import argparse, csv, math, re, statistics, unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = "impresso-project/OCR-quality-assessment-unigram"
VERSION = "2.0.0"
LANG = "fr"
SOURCES = {"tesseract": "text.txt", "pero": "pero.txt"}
LABELS = {"tesseract": "Tesseract", "pero": "PERO"}
COLORS = {"tesseract": "#2a78d6", "pero": "#eb6834"}

SCORES_CSV = ROOT / "ocrqa_scores.csv"
SUMMARY_CSV = ROOT / "ocrqa_summary.csv"
REPORT_MD = ROOT / "ocrqa_report.md"
CHART_PNG = ROOT / "ocrqa_comparison.png"

# Normalisation from the model card: lowercase, punctuation -> space, digits -> 0
PUNCT = "„•<>!\"#%&'’()*,./:;?[]\\~_{}\xa1\xab\xb7\xbb\xbf—^`¦§£=-"
TABLE = str.maketrans({c: " " for c in PUNCT} | {d: "0" for d in "0123456789"})

FIELDS = ["issue", "date"] + [f"{s}_{k}" for s in SOURCES
                              for k in ("ocrqa", "known_types", "types", "tokens")] + ["delta_pero_minus_tess"]


# ---------------------------------------------------------------- scoring
def dehyphen(text):
    # rejoin words split across lines: "inter-\nrogatoire" -> "interrogatoire"
    return re.sub(r"(\w)[-¬]\s*\n\s*(\w)", r"\1\2", text)


def tokenize(text):
    return unicodedata.normalize("NFKC", text).lower().translate(TABLE).split()


def score_text(text, known):
    toks = tokenize(text)
    types = set(toks)
    n_known = sum(t in known for t in types)
    return {"ocrqa": round(n_known / len(types), 3) if types else None,
            "known_types": n_known, "types": len(types), "tokens": len(toks)}


def issue_date(name):
    m = re.search(r"\d{4}-\d{2}-\d{2}", name)
    return m.group(0) if m else ""


def score_all(dehyphenate):
    from huggingface_hub import hf_hub_download
    from pybloomfilter import BloomFilter

    known = BloomFilter.open(hf_hub_download(MODEL, f"ocrqa-wp_v{VERSION}-{LANG}.bloom"), "r")
    rows = []
    for folder in sorted(ROOT.glob("depechetoulouse-*")):
        row = {"issue": folder.name, "date": issue_date(folder.name)}
        for src, fname in SOURCES.items():
            p = folder / fname
            if not p.exists():
                continue
            text = p.read_text(encoding="utf-8")
            if dehyphenate:
                text = dehyphen(text)
            for k, v in score_text(text, known).items():
                row[f"{src}_{k}"] = v
        if row.get("tesseract_ocrqa") is not None and row.get("pero_ocrqa") is not None:
            row["delta_pero_minus_tess"] = round(row["pero_ocrqa"] - row["tesseract_ocrqa"], 3)
        rows.append(row)
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def load_scores():
    """Read ocrqa_scores.csv (also the older format without date / known_types columns)."""
    rows = []
    with SCORES_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {"issue": r["issue"], "date": r.get("date") or issue_date(r["issue"])}
            for src in SOURCES:
                if not r.get(f"{src}_ocrqa"):
                    continue
                q, ty, tok = float(r[f"{src}_ocrqa"]), int(r[f"{src}_types"]), int(r[f"{src}_tokens"])
                kn = r.get(f"{src}_known_types")
                row.update({f"{src}_ocrqa": q, f"{src}_types": ty, f"{src}_tokens": tok,
                            f"{src}_known_types": int(kn) if kn else round(q * ty)})
            if f"tesseract_ocrqa" in row and "pero_ocrqa" in row:
                row["delta_pero_minus_tess"] = round(row["pero_ocrqa"] - row["tesseract_ocrqa"], 3)
            rows.append(row)
    return rows


# ---------------------------------------------------------------- summary
def sign_test_p(wins, losses):
    """Two-sided exact sign test (ties dropped): how likely is a split this lopsided by chance?"""
    n = wins + losses
    if n == 0:
        return None
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize(rows):
    paired = [r for r in rows if "delta_pero_minus_tess" in r]
    summary = []
    for src in SOURCES:
        rs = [r for r in paired if f"{src}_ocrqa" in r]
        if not rs:
            continue
        q = [r[f"{src}_ocrqa"] for r in rs]
        types = sum(r[f"{src}_types"] for r in rs)
        tokens = sum(r[f"{src}_tokens"] for r in rs)
        summary.append({
            "engine": LABELS[src], "issues": len(rs),
            "mean_ocrqa": round(statistics.mean(q), 3),
            "median_ocrqa": round(statistics.median(q), 3),
            "sd_ocrqa": round(statistics.stdev(q), 3) if len(q) > 1 else 0.0,
            "min_ocrqa": min(q), "max_ocrqa": max(q),
            "pooled_ocrqa": round(sum(r[f"{src}_known_types"] for r in rs) / types, 3),
            "total_tokens": tokens, "total_types": types,
            "mean_tokens_per_issue": round(tokens / len(rs)),
            "mean_types_per_issue": round(types / len(rs)),
        })
    d = [r["delta_pero_minus_tess"] for r in paired]
    wins, losses = sum(x > 0 for x in d), sum(x < 0 for x in d)
    paired_stats = {
        "n": len(d), "mean_delta": statistics.mean(d) if d else None,
        "median_delta": statistics.median(d) if d else None,
        "min_delta": min(d) if d else None, "max_delta": max(d) if d else None,
        "pero_better": wins, "tesseract_better": losses, "ties": len(d) - wins - losses,
        "sign_test_p": sign_test_p(wins, losses),
    }
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    return summary, paired_stats


# ---------------------------------------------------------------- chart
def make_chart(rows, summary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -> no chart (pip install matplotlib)")
        return False
    rs = sorted((r for r in rows if "delta_pero_minus_tess" in r), key=lambda r: r["date"])
    y = list(range(len(rs)))
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    fig, ax = plt.subplots(figsize=(7.5, 0.38 * len(rs) + 1.6), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for i, r in zip(y, rs):
        ax.plot([r["tesseract_ocrqa"], r["pero_ocrqa"]], [i, i], color=grid, lw=2, zorder=1)
    means = {s["engine"]: s["mean_ocrqa"] for s in summary}
    for src in SOURCES:
        ax.scatter([r[f"{src}_ocrqa"] for r in rs], y, s=55, color=COLORS[src],
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=3,
                   label=f"{LABELS[src]} (mean {means[LABELS[src]]:.3f})")
        ax.axvline(means[LABELS[src]], color=COLORS[src], lw=1, ls=(0, (3, 3)), zorder=0)
    # gain column in the right margin, clear of the dots and the mean lines
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(1.02, -0.9, "Gain", transform=tr, ha="left", va="center", fontsize=8, color=ink)
    for i, r in zip(y, rs):
        ax.text(1.02, i, f"{r['delta_pero_minus_tess']:+.3f}", transform=tr,
                ha="left", va="center", fontsize=8, color=muted)
    ax.set_yticks(y, [r["date"] for r in rs], fontsize=8, color=ink)
    ax.invert_yaxis()
    lo = min(min(r["tesseract_ocrqa"], r["pero_ocrqa"]) for r in rs)
    ax.set_xlim(math.floor(lo * 20) / 20 - 0.02, 1.0)
    ax.set_xlabel("OCRQA: share of distinct words found in the French lexicon (higher = cleaner)",
                  fontsize=8.5, color=ink)
    ax.tick_params(axis="x", labelsize=8, colors=muted)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=grid, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_title("OCR quality per issue — La Dépêche (Toulouse), 1912, Le Secret du Titan",
                 fontsize=10, color=ink, loc="left", pad=26)
    ax.legend(loc="lower left", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0, 1.0),
              handletextpad=0.3, columnspacing=1.5, borderaxespad=0.2)
    fig.tight_layout()
    fig.subplots_adjust(right=0.9)
    fig.savefig(CHART_PNG, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


# ---------------------------------------------------------------- report
def make_report(rows, summary, ps, chart_ok):
    s = {x["engine"]: x for x in summary}
    t, p = s["Tesseract"], s["PERO"]
    pval = ps["sign_test_p"]
    ptxt = f"{pval:.5f}" if pval is not None and pval >= 1e-4 else "< 0.0001"
    L = []
    L.append("# OCR quality: Tesseract vs PERO OCR\n")
    L.append(f"Corpus: {ps['n']} issues of *La Dépêche* (Toulouse), 1912, feuilleton *Le Secret du Titan* "
             f"(X. de Montépin). Each issue = the same column crops, transcribed by both engines.\n")
    L.append("## Headline\n")
    L.append(f"- **Mean OCRQA: Tesseract {t['mean_ocrqa']:.3f} → PERO {p['mean_ocrqa']:.3f}** "
             f"(mean gain {ps['mean_delta']:+.3f}, i.e. {ps['mean_delta']*100:+.1f} points).")
    L.append(f"- PERO scores higher on **{ps['pero_better']} of {ps['n']} issues** "
             f"(Tesseract higher on {ps['tesseract_better']}, ties {ps['ties']}); "
             f"gain per issue ranges from {ps['min_delta']:+.3f} to {ps['max_delta']:+.3f}. "
             f"Exact sign test, two-sided: p = {ptxt}.")
    L.append(f"- Both engines read about the same amount of text "
             f"({t['mean_tokens_per_issue']} vs {p['mean_tokens_per_issue']} words per issue), "
             f"so PERO's higher score is not explained by dropping hard passages.")
    L.append(f"- PERO produces fewer distinct word forms ({t['mean_types_per_issue']} → "
             f"{p['mean_types_per_issue']} per issue): fewer garbled variants of the same words.\n")
    if chart_ok:
        L.append(f"![OCRQA per issue]({CHART_PNG.name})\n")
    L.append("## Summary table\n")
    L.append("| | Tesseract | PERO |")
    L.append("|---|---:|---:|")
    for key, lab, fmt in [("mean_ocrqa", "Mean OCRQA (average of issues)", "{:.3f}"),
                          ("median_ocrqa", "Median OCRQA", "{:.3f}"),
                          ("sd_ocrqa", "Std. deviation across issues", "{:.3f}"),
                          ("min_ocrqa", "Worst issue", "{:.3f}"),
                          ("max_ocrqa", "Best issue", "{:.3f}"),
                          ("pooled_ocrqa", "Pooled OCRQA (all issues' types summed)", "{:.3f}"),
                          ("mean_tokens_per_issue", "Words (tokens) per issue", "{:,}"),
                          ("mean_types_per_issue", "Distinct words (types) per issue", "{:,}"),
                          ("total_tokens", "Total tokens", "{:,}"),
                          ("total_types", "Total types (summed per issue)", "{:,}")]:
        L.append(f"| {lab} | {fmt.format(t[key])} | {fmt.format(p[key])} |")
    L.append("")
    L.append("## Per issue\n")
    L.append("| Issue date | Tesseract OCRQA | PERO OCRQA | Gain | Tesseract tokens / types | PERO tokens / types |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r in sorted((r for r in rows if "delta_pero_minus_tess" in r), key=lambda r: r["date"]):
        L.append(f"| {r['date']} | {r['tesseract_ocrqa']:.3f} | {r['pero_ocrqa']:.3f} | "
                 f"{r['delta_pero_minus_tess']:+.3f} | {r['tesseract_tokens']:,} / {r['tesseract_types']:,} | "
                 f"{r['pero_tokens']:,} / {r['pero_types']:,} |")
    L.append("")
    L.append("## What the columns mean\n")
    L.append("Before counting, each text is normalised the way the Impresso model expects: lowercased, "
             "punctuation replaced by spaces, digits replaced by 0, then split on spaces.\n")
    L.append("- **tokens** — every word the engine produced, repeats included. If *de* occurs 80 times, "
             "that is 80 tokens. It measures *how much* text the engine read.")
    L.append("- **types** — the *distinct* words among those tokens. The 80 *de* count once. "
             "Each misspelling (*interrogutoire*, *interrogatoirc*) is a new type, so noisy OCR "
             "inflates the type count.")
    L.append("- **known_types** — the types found in the French lexicon.")
    L.append("- **ocrqa** = known_types / types — the share of distinct words that are real French "
             "words. 1.0 = every distinct word is in the lexicon.")
    L.append("- **Gain** (delta) = PERO OCRQA − Tesseract OCRQA on the same issue.")
    L.append("- **Mean OCRQA** averages the per-issue scores (each issue weighs the same); "
             "**pooled OCRQA** divides all known types by all types, so longer issues weigh more.\n")
    L.append("Toy example: *« le lecteur lit le journal journl »* → 6 tokens, 5 types "
             "(le, lecteur, lit, journal, journl), 4 known → OCRQA = 4/5 = 0.80.\n")
    L.append("## Caveats\n")
    L.append("- OCRQA needs no reference transcription, so it is **not a character error rate**: "
             "a misreading that yields another real word (*mère* → *mer*) counts as correct, while "
             "proper names and old spellings absent from Wikipedia count as errors. Both engines face "
             "the same text, so the comparison is fair; the absolute values are approximate.")
    L.append("- It ignores missing text — hence the token counts above, which are similar for both engines.")
    L.append("- The lexicon is a Bloom filter: a small share of garbage strings are wrongly accepted, "
             "which lifts both engines' scores slightly and equally.")
    L.append("- Line-end hyphenation is left as is (Tesseract lost most hyphens), so word fragments "
             "penalise both engines alike.")
    REPORT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dehyphen", action="store_true")
    ap.add_argument("--report-only", action="store_true",
                    help="skip scoring; rebuild summary/report/chart from ocrqa_scores.csv")
    a = ap.parse_args()

    rows = load_scores() if a.report_only else score_all(a.dehyphen)
    summary, ps = summarize(rows)
    chart_ok = make_chart(rows, summary)
    make_report(rows, summary, ps, chart_ok)

    for s in summary:
        print(f"{s['engine']:9s} n={s['issues']:2d}  mean={s['mean_ocrqa']:.3f}  median={s['median_ocrqa']:.3f}  "
              f"sd={s['sd_ocrqa']:.3f}  tokens/issue={s['mean_tokens_per_issue']}  types/issue={s['mean_types_per_issue']}")
    print(f"paired gain (PERO - Tesseract): mean {ps['mean_delta']:+.3f}, "
          f"PERO higher on {ps['pero_better']}/{ps['n']} issues, sign-test p = {ps['sign_test_p']:.2g}")
    for f in (SCORES_CSV, SUMMARY_CSV, REPORT_MD) + ((CHART_PNG,) if chart_ok else ()):
        print(f"-> {f.name}")


if __name__ == "__main__":
    main()
