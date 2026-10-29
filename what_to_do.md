
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

### ✅ Confirmed FR–DE pair: Montépin, *Le Secret du Titan*

| Side | Title | Paper | Run | Installments | First ID |
| --- | --- | --- | --- | --- | --- |
| FR | *Le Secret du Titan* | La Dépêche (Toulouse) | 1912-06-16 → ~Oct 1912, numbered N. 1 … | 88 | `depechetoulouse-1912-06-16-a-i0030` |
| DE | *Das Geheimniß* ("frei übersetzt von N. B.") | Luxemburger Wort | 1891-07-27 → 1891-12-22 | 118 | search `title="Geheimniß"`, `media_source_id="luxwort"` |

Notes: German (1891) is 21 years *older* than this French run, so the French original's first edition is earlier — this is a reprint. "Frei übersetzt" → expect cuts and paraphrase: a good hard test for embeddings vs. names.

Also found: *Les Filles de bronze* FR in Le Figaro (Oct 1879 – Feb 1880) and La Dépêche (Oct 1889); no German version found yet.