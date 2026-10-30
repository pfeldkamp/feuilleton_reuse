# Shared feature code, so every script computes MFW, passim and similarities the same way.
import re
from collections import Counter

import numpy as np
from scipy.stats import spearmanr


def words(text):
    return re.findall(r"[^\W\d_]+", text.lower())


# --- 100 most frequent words -----------------------------------------------------

def mfw_z(texts_a, texts_b, n=100):
    """z-scored relative frequencies of the n most frequent words of a and b together.

    Returns (z_a, z_b, word_list); both sides use the same words and the same mean/std.
    """
    counts_a = [Counter(words(t)) for t in texts_a]
    counts_b = [Counter(words(t)) for t in texts_b]
    total = sum((c for c in counts_a + counts_b), Counter())
    mfw = [w for w, _ in total.most_common(n)]

    def freqs(counts):
        return np.array([[c[w] / max(sum(c.values()), 1) for w in mfw] for c in counts])

    fa, fb = freqs(counts_a), freqs(counts_b)
    stacked = np.vstack([fa, fb])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0)
    std[std == 0] = 1
    return (fa - mean) / std, (fb - mean) / std, mfw


# --- passim-style score ----------------------------------------------------------
# passim's first step (shared rare word n-grams) in plain Python; passim itself needs
# Java + Spark and then also aligns passages (Smith-Waterman), which we skip.

def ngrams(text, n=5):
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def passim_matrix(texts_a, texts_b, n=5, max_df=5):
    """score[i, j] = share of a[i]'s rare n-grams that also occur in b[j].

    Rare = found in at most max_df installments of a and b together (drops headers, set phrases).
    """
    grams_a = [ngrams(t, n) for t in texts_a]
    grams_b = [ngrams(t, n) for t in texts_b]
    df = Counter(g for gs in grams_a + grams_b for g in gs)
    grams_a = [{g for g in gs if df[g] <= max_df} for gs in grams_a]
    grams_b = [{g for g in gs if df[g] <= max_df} for gs in grams_b]
    return np.array([[len(ga & gb) / len(ga) if ga else 0.0 for gb in grams_b] for ga in grams_a])


# --- similarity and scoring ------------------------------------------------------

def cosine_matrix(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def score(sim, expected, tolerance=0, best=None):
    """How well does each row's best match follow the novel?

    progression: Spearman correlation between where each installment really is (expected)
                 and where its best match is; 1 = everything found in order, 0 = random.
    top1:        share of rows whose best match is the expected one (+- tolerance).
    separation:  how far the right match stands out from all the other candidates, in standard
                 deviations (median over rows); keeps falling smoothly when top-1 is still 1.0.
    median_error: how many installments the best match is off (median).
    """
    best = sim.argmax(axis=1) if best is None else np.asarray(best)  # or a given mapping
    order = (-sim).argsort(axis=1, kind="stable")
    rank = [min(np.where(np.abs(order[i] - expected[i]) <= tolerance)[0]) + 1 for i in range(len(sim))]
    separation = []
    for i, row in enumerate(sim):
        window = np.abs(np.arange(len(row)) - expected[i]) <= tolerance
        others = row[~window]
        separation.append((row[window].max() - others.mean()) / (others.std() or 1))
    return {
        "progression": float(spearmanr(expected, best).correlation),
        "top1": float(np.mean(np.abs(best - expected) <= tolerance)),
        "separation": float(np.median(separation)),
        "median_error": float(np.median(np.abs(best - expected))),
        "median_rank": float(np.median(rank)),
        "chance_top1": float((2 * tolerance + 1) / sim.shape[1]),
        "rows_with_no_signal": int((sim.max(axis=1) == 0).sum()),
    }


# --- named entities ----------------------------------------------------------------
# Names are kept as written (no Pierre <-> Peter mapping). Two units:
#   full names:  "pierre landry" and "peter landry" are different names
#   name words:  the names split into words, so "landry" matches across languages

def name_bag(entities, unit="name words", types=("pers", "loc")):
    bag = Counter()
    for e in entities:
        if e["type"] not in types:
            continue
        name = re.sub(r"\s+", " ", (e.get("name") or e["surface"]).lower()).strip()
        if unit == "full names":
            bag[name] += 1
        else:
            bag.update(w for w in words(name) if len(w) > 1)
    return bag


def tfidf_cosine(bags_a, bags_b):
    """Cosine similarity of TF-IDF vectors; rare names (in few installments) weigh most."""
    bags = bags_a + bags_b
    vocab = sorted({w for b in bags for w in b})
    index = {w: i for i, w in enumerate(vocab)}
    df = Counter(w for b in bags for w in b)
    idf = np.array([np.log(len(bags) / df[w]) for w in vocab])

    def matrix(bs):
        m = np.zeros((len(bs), len(vocab)))
        for i, b in enumerate(bs):
            for w, c in b.items():
                m[i, index[w]] = c
        m *= idf
        m[m.sum(axis=1) == 0, 0] = 1e-9  # avoid dividing by zero for an installment without names
        return m

    return cosine_matrix(matrix(bags_a), matrix(bags_b))


def combine(*sims):
    """Average of similarity matrices after z-scoring each row, so no feature's scale dominates."""
    z = [(s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-12) for s in sims]
    return sum(z) / len(z)


# --- in-order alignment -------------------------------------------------------------

def align_in_order(sim):
    """Give each row one column, with columns never going backwards, maximising total similarity.

    Rows may share a column and columns may be skipped. Dynamic programming over z-scored rows.
    Note: the result is in order by construction, so judge it by top-1 / median error;
    progression (Spearman) is high for any in-order path.
    """
    z = combine(sim)
    n_rows, n_cols = z.shape
    total = np.zeros_like(z)
    came_from = np.zeros_like(z, dtype=int)
    total[0] = z[0]
    for i in range(1, n_rows):
        best_so_far = np.maximum.accumulate(total[i - 1])
        # the column where that best running total was reached
        came_from[i] = np.where(total[i - 1] == best_so_far, np.arange(n_cols), 0)
        came_from[i] = np.maximum.accumulate(came_from[i])
        total[i] = z[i] + best_so_far
    path = [int(total[-1].argmax())]
    for i in range(n_rows - 1, 0, -1):
        path.append(int(came_from[i, path[-1]]))
    return path[::-1]
