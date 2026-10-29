
# CONCEPT

We'd like to trace how a few romans-feuilletons (Sue, Dumas) travelled from Paris to other, Swiss and Luxembourgish papers
Concept: check whether embeddings find Les Mystères de Paris again across its French and German newspaper reprints.

## Data
Get installments of Les Mystères de Paris (1842–43) from **Journal des débats** (dates in Rez-de-chaussée PDFs).

- google translate the French installments to German and English for comparison.
- Find the same chapters in German reprints in Impresso.
- Chunk the French and German texts the same way.
- Embed (GTE + high MTEB score one) and feat extract (NER, Triplets)

## TEST 1
- do the embedding cosims and PASSIM ranking correlate when looking at the French original?

## TEST 2
- Does original installment 1 match its google translated counterparts?

## TEST 3
- Does original installment 1 match its German reprint counterpart? (and so on)

## Output
a results table for GTE vs NER vs POSITION SIGNAL vs GTE + NER and a short write-up.



    
# DATA

## ✅ Confirmed FR–DE pair: Montépin, *Le Secret du Titan*

| Side | Title | Paper | Run | Installments | First ID |
| --- | --- | --- | --- | --- | --- |
| FR | *Le Secret du Titan* | La Dépêche (Toulouse) | 1912-06-16 → ~Oct 1912, numbered N. 1 … | 88 | `depechetoulouse-1912-06-16-a-i0030` |
| DE | *Das Geheimniß* ("frei übersetzt von N. B.") | Luxemburger Wort | 1891-07-27 → 1891-12-22 | 118 | search `title="Geheimniß"`, `media_source_id="luxwort"` |

Notes: German (1891) is 21 years *older* than this French run, so the French original's first edition is earlier — this is a reprint. "Frei übersetzt" → expect cuts and paraphrase: a good hard test for embeddings vs. names.

Also found: *Les Filles de bronze* FR in Le Figaro (Oct 1879 – Feb 1880) and La Dépêche (Oct 1889); no German version found yet.

## added the OCR of scans that have "texte illegible"
- 14 installments re-OCR'd with Tesseract (`reocr_broken_fr.py`), flagged `"ocr_source": "tesseract"` in fr.jsonl

| Installment | Status |
| --- | --- |
| N.8 | **the only real gap**: no 23 or 24 June 1912 issue in Impresso; the story jumps from 22 June (N.7) to 25 June (N.9, Pierre Landry mid-confession) |
| N.58 | found by hand (`depechetoulouse-1912-09-01-a-i0108`, in `missing_fr_ids.txt`), OCR empty → re-OCR'd; text runs on from N.57 to N.59 |
| N.81 | numbering skip by the paper: 27 Sep ("N. 80, 26 Septembre") runs straight on into 28 Sep (N.82), no text lost |
| N.12, N.20, N.21 | printed twice by the paper (29+30 June, 14+15 July, 16+18 July): different text each day, runs on |
| N.88 | 4 Oct printed as "N. 87" again, but runs on from 3 Oct and into 5 Oct (N.89) |
| N.42, 48, 53 | present: OCR misread the headers ("N. 12" on 15 Aug, "N. 43" on 22 Aug, "N. 27 Août" on 27 Aug) |

Checked by reading the end of every installment against the start of the next (29 Oct): all 91 transitions run on except 22 → 25 June.
So the novel has 93 installments (labelled up to N.91 because of 3 repeats and 1 skip); we have 92, missing only N.8.