# %%

from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy

# %%

CWD = Path(__file__).parent
data = CWD / "secret_du_titan" / "chunked"
de_file = "chunked_de.jsonl"
fr_file = "chunked_fr.jsonl"

# %%

models = ['impresso-project/histlux-gte-multilingual-base', 'impresso-project/OCR-diversely-robust-gte-multilingual-base']

for name in models:
    model = SentenceTransformer(name, trust_remote_code=True)

    sav_dict = {}

    for file in [de_file, fr_file]:
        sentences = pd.read_json(data / file, lines=True)['text'].tolist()
        embeddings = model.encode(sentences)
        sav_dict[file] = embeddings.tolist()
        with open(CWD / f"{file}_{name}_embeddings.json", "w") as f:
            import json
            json.dump(sav_dict[file], f)
            print(f"Saved embeddings for {file} using model {name}")



# %%