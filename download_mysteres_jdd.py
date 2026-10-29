# Download the 150 installments of Sue's "Les Mystères de Paris" from the
# Journal des débats (19 Jun 1842 - 15 Oct 1843) out of Impresso.
#
# Why this is not a simple search: Impresso splits each issue into items that mix
# news and feuilleton (the feuilleton sits in the bottom band of the page, the
# "rez-de-chaussée", and some of its columns are glued onto news items above).
# So we cut by position on the page instead of by item:
#   1. dates + pages of each installment come from the Rez-de-chaussée list (PDF)
#   2. for each date, list all items in the issue
#   3. find where the feuilleton band starts on each page: the height (between
#      55% and 80% of the page) where at least 3 of the 4 columns start a new
#      region at the same level; keep every text region that starts below it
#   4. put the regions in reading order (page, column, top to bottom) and join them
#
# Output: mysteres/fr_jdd.jsonl (one installment per line) and mysteres/fr_jdd.txt
# Re-running skips installments already downloaded.
import json
import re
import time
import urllib.request
from pathlib import Path

from impresso import connect, DateRange
from pypdf import PdfReader

HERE = Path(__file__).parent
PDF = HERE.parent / "rez-de-chausee" / "mystere_debats.pdf"
OUT = HERE / "mysteres"
COLUMN_WIDTH = 1075  # the pages have 4 columns of ~1075 px
FALLBACK_BAND_TOP = 0.67  # share of the page height, if the band line is not found

MONTHS = {m: i for i, m in enumerate(
    ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
     "août", "septembre", "octobre", "novembre", "décembre"], 1)}

client = connect(public_api_url="https://dev.impresso-project.ch/public-api/v1")


def installments_from_pdf():
    """[(number, 'YYYY-MM-DD', [pages])] from the Rez-de-chaussée list."""
    text = "\n".join(p.extract_text() for p in PdfReader(PDF).pages)
    # drop the page header/footer, which can fall in the middle of an entry
    text = re.sub(r"\n\d+ / \d+\n|Le Rez-de-chaussée - Fiche de l'Oeuvre\n", "\n", text)
    found = re.findall(r"Feuilleton (\d+)\s+Date de publication : (\d+) (\w+) (\d{4})\s+"
                       r"Pages du journal : ([\d-]+)", text)
    return [(int(n), f"{y}-{MONTHS[m]:02d}-{int(d):02d}", [int(p) for p in pages.split("-")])
            for n, d, m, y, pages in found]


def retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            if i == tries - 1:
                raise
            print(f"  retry {i + 1} after {type(e).__name__}", flush=True)
            time.sleep(5 * (i + 1))


page_height = {}


def height(page):
    url = page["iiif"]["manifestUrl"]
    if url not in page_height:
        page_height[url] = retry(lambda: json.load(urllib.request.urlopen(url))["height"])
    return page_height[url]


def column(box):
    return round(box[0] / COLUMN_WIDTH)


def find_band_tops(issue, pages):
    """{page number: y where the feuilleton band starts}, from all boxes in the issue."""
    boxes = {p: [] for p in pages}
    page_info = {}
    for row_pages in issue["facsimile.pages"]:
        for page in row_pages:
            if page["number"] in boxes:
                boxes[page["number"]] += page["regionCoordinates"]
                page_info[page["number"]] = page
    tops = {}
    for number, page in page_info.items():
        h = height(page)
        starts = sorted((b[1], column(b)) for b in boxes[number] if 0.55 * h < b[1] < 0.8 * h)
        tops[number] = FALLBACK_BAND_TOP * h
        for y0, _ in starts:
            if len({c for y, c in starts if y0 <= y <= y0 + 100}) >= 3:
                tops[number] = y0 - 30
                break
    return tops


def band_regions(item_id, band_tops):
    """Text regions of one item that lie in the feuilleton band of the given pages."""
    raw = retry(lambda: client.content_items.get(item_id).raw)
    text, fac = raw["text"]["content"], raw["facsimile"]
    cuts = [0] + fac["regionBreaks"] + [len(text)]
    regions, k = [], 0
    for page in fac["pages"]:
        for box in page["regionCoordinates"]:
            if page["number"] in band_tops and box[1] >= band_tops[page["number"]]:
                regions.append({"page": page["number"], "x": box[0], "y": box[1],
                                "text": text[cuts[k]:cuts[k + 1]].strip(), "item": item_id})
            k += 1
    return regions


OUT.mkdir(exist_ok=True)
path = OUT / "fr_jdd.jsonl"
done = {}
if path.exists():
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        done[rec["n"]] = rec

todo = installments_from_pdf()
print(f"{len(todo)} installments in the list, {len(done)} already downloaded", flush=True)

for n, date, pages in todo:
    if n in done:
        continue
    issue = retry(lambda: client.search.find(media_source_id="jdpl",
                                             date_range=DateRange(date, date), limit=100)).df
    band_tops = find_band_tops(issue, pages)
    # only fetch items that have at least one box in the band of a feuilleton page
    candidates = [uid for uid, row in issue.iterrows()
                  if any(p["number"] in band_tops and b[1] >= band_tops[p["number"]]
                         for p in row["facsimile.pages"] for b in p["regionCoordinates"])]
    regions = [r for uid in candidates for r in band_regions(uid, band_tops)]
    # reading order: page, then column, then top to bottom
    regions.sort(key=lambda r: (r["page"], column([r["x"]]), r["y"]))
    rec = {"n": n, "date": date, "pages": pages,
           "band_tops": {p: round(y) for p, y in band_tops.items()},
           "items": sorted({r["item"] for r in regions}),
           "n_regions": len(regions),
           "text": "\n".join(r["text"] for r in regions)}
    done[n] = rec
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  {n:3d} {date} pages {pages}: {len(regions)} regions, {len(rec['text'])} chars", flush=True)

recs = sorted(done.values(), key=lambda r: r["n"])
path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
(OUT / "fr_jdd.txt").write_text(
    "\n\n".join(f"=== Feuilleton {r['n']} {r['date']} ===\n{r['text']}" for r in recs))
print(f"saved {len(recs)} installments; empty: {[r['n'] for r in recs if not r['text']]}")
