# Re-OCR the French installments of "Le Secret du Titan" whose Impresso OCR failed
# (text is only the header + "[texte illisible]").
#
# For each broken installment:
#   1. ask the Impresso API where the article sits on the page (one box per column)
#   2. download each box as an image from the BnF IIIF server (Gallica scans)
#   3. OCR each box with Tesseract (French) and join the columns in order
#
# Output:
#   secret_du_titan/reocr/<id>/r01.jpg, r02.jpg, ...  the cropped columns (check by eye)
#   secret_du_titan/reocr/<id>/text.txt               the new OCR text
#   secret_du_titan/fr.jsonl, fr.txt                  updated in place:
#       every record gets "ocr_source": "impresso" or "tesseract"
#       re-OCR'd records keep the old text in "impresso_text"
#
# Needs: brew install tesseract + fra.traineddata in the tessdata folder.
# Re-running skips installments that were already re-OCR'd.
import json
import subprocess
import time
import urllib.request
from pathlib import Path

from impresso import connect

DATA = Path(__file__).parent / "secret_du_titan"
REOCR = DATA / "reocr"
BROKEN_MAX_CHARS = 400  # broken installments are ~200 chars; real ones are ~10,000

client = connect(public_api_url="https://dev.impresso-project.ch/public-api/v1")


def column_boxes(item_id):
    """Return (iiif_base_url, [x, y, w, h]) for every column box of the article."""
    raw = client.content_items.get(item_id).raw
    boxes = []
    for page in raw["facsimile"]["pages"]:
        base = page["iiif"]["manifestUrl"].removesuffix("/info.json")
        for box in page["regionCoordinates"]:
            boxes.append((base, box))
    return boxes


def download(url, path):
    if not path.exists():
        urllib.request.urlretrieve(url, path)
        time.sleep(1)  # be polite to the BnF server


def ocr(image_path):
    # --psm 4: one column of text of variable sizes
    out = subprocess.run(["tesseract", str(image_path), "-", "-l", "fra", "--psm", "4"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


records = [json.loads(line) for line in (DATA / "fr.jsonl").read_text().splitlines()]

for rec in records:
    if rec.get("ocr_source") == "tesseract":
        continue
    rec["ocr_source"] = "impresso"
    if len(rec["text"]) >= BROKEN_MAX_CHARS:
        continue

    folder = REOCR / rec["id"]
    folder.mkdir(parents=True, exist_ok=True)
    columns = []
    for k, (base, (x, y, w, h)) in enumerate(column_boxes(rec["id"]), 1):
        img = folder / f"r{k:02d}.jpg"
        download(f"{base}/{x},{y},{w},{h}/max/0/default.jpg", img)
        columns.append(ocr(img))
    text = "\n\n".join(columns)
    text = text.replace("-\n", "")  # rejoin words hyphenated at line ends
    (folder / "text.txt").write_text(text)

    rec["impresso_text"] = rec["text"]
    rec["text"] = text
    rec["ocr_source"] = "tesseract"
    print(f"{rec['date']} {rec['id']}: {len(columns)} columns, {len(text)} chars", flush=True)

(DATA / "fr.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
(DATA / "fr.txt").write_text(
    "\n\n".join(f"=== {r['date']} {r['id']} ===\n{r['text']}" for r in records)
)
print("done:", sum(r["ocr_source"] == "tesseract" for r in records), "installments re-OCR'd")
