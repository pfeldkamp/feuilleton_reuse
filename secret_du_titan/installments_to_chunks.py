# %%
from pathlib import Path
import pandas as pd

# %%

CWD = Path(__file__).parent
data = CWD / "secret_du_titan"
de_file = "de.jsonl"
fr_file = "fr.jsonl"

stats_df = pd.DataFrame(columns=["file", "n_installments", "avg_length_tokens", "avg_length_words", "n_words"])

for file in [de_file, fr_file]:
    print(f"Processing {file}")
    text = pd.read_json(data / file, lines=True)
    n_installments = len(text)
    avg_length = text['text'].str.split().str.len().mean()
    avg_length_tokens = text['text'].str.len().mean()
    n_words = text['text'].str.split().str.len().sum()
    stats_df = pd.concat([stats_df, pd.DataFrame([{
        "file": file,
        "n_installments": n_installments,
        "avg_length_chars": avg_length_tokens,
        "avg_length_words": avg_length,
        "n_words": n_words
    }])], ignore_index=True)

print(stats_df)

# %%
text.columns
# %%

# print the ids of all installments with "texte illegible"

# get all columns
text = pd.read_json(data / fr_file, lines=True)
print(f"Columns in {file}: {text.columns.tolist()}")
# get "id" col for row where "impresso_text" contains "texte illisible"
illegible_ids = text.loc[text['impresso_text'].str.contains("texte illisible", na=False), 'id'].tolist()
print(f"Illegible installments in {fr_file}: {illegible_ids}")

# of how many retrieved document was this the case?
print(f"Number of illegible installments in {fr_file}: {len(illegible_ids)}")
print(f"Percentage of illegible installments in {fr_file}: {len(illegible_ids) / len(text) * 100:.2f}%")

# %%



# %%

# concat all text and split it into 100 chunks

# %%



