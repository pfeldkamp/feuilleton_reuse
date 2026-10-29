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
model = SentenceTransformer('impresso-project/histlux-gte-multilingual-base', trust_remote_code=True)

sav_dict = {}

for file in [de_file, fr_file]:
    sentences = pd.read_json(data / file, lines=True)['text'].tolist()
    embeddings = model.encode(sentences)
    # save as json
    sav_dict[file] = embeddings.tolist()
    # save to json file
    with open(CWD / f"{file}_embeddings.json", "w") as f:
        import json
        json.dump(sav_dict[file], f)


# %%