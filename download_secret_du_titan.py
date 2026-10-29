# Download every installment of Montépin's "Le Secret du Titan" (FR, La Dépêche 1912)
# and its German translation "Das Geheimniß" (DE, Luxemburger Wort 1891).
# Output: data/secret_du_titan/{fr,de}.jsonl (one installment per line, date order)
#         data/secret_du_titan/{fr,de}.txt   (all installments concatenated)
# Re-running skips installments already downloaded.
import json
import time
from pathlib import Path

from impresso import connect, DateRange, AND

OUT = Path(__file__).parent / "data" / "secret_du_titan"
OUT.mkdir(parents=True, exist_ok=True)

client = connect(public_api_url="https://dev.impresso-project.ch/public-api/v1")

SIDES = {
    "fr": dict(title=AND("Secret", "Titan"), language="fr", media_source_id="depechetoulouse",
               date_range=DateRange("1912-01-01", "1913-12-31")),
    "de": dict(title="Geheimniß", language="de", media_source_id="luxwort",
               date_range=DateRange("1891-01-01", "1892-06-30")),
}


def retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} after {type(e).__name__}", flush=True)
            time.sleep(5 * (i + 1))


def list_ids(query):
    rows, offset = [], 0
    while True:
        r = retry(lambda: client.search.find(**query, order_by="date", limit=100, offset=offset))
        df = r.df
        if df.empty:
            break
        for uid, x in df.iterrows():
            rows.append({"id": uid, "date": x["meta.date"][:10], "title": x.get("text.title")})
        offset += len(df)
        if offset >= r.total:
            break
    return rows


for lang, query in SIDES.items():
    path = OUT / f"{lang}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec

    rows = list_ids(query)
    print(f"{lang}: {len(rows)} installments listed, {len(done)} already downloaded", flush=True)

    for n, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        item = retry(lambda: client.content_items.get(row["id"]))
        raw = item.raw if hasattr(item, "raw") else item
        row["text"] = raw.get("text", {}).get("content", "")
        row["ocr_quality"] = raw.get("semanticEnrichments", {}).get("ocrQuality")
        done[row["id"]] = row
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  {lang} {n}/{len(rows)} {row['date']} {len(row['text'])} chars", flush=True)
        time.sleep(0.5)

    # Rewrite in date order and build the concatenated text
    recs = sorted(done.values(), key=lambda r: (r["date"], r["id"]))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    (OUT / f"{lang}.txt").write_text(
        "\n\n".join(f"=== {r['date']} {r['id']} ===\n{r['text']}" for r in recs)
    )
    print(f"{lang}: saved {len(recs)} installments", flush=True)
