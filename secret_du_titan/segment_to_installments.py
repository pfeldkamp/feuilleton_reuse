# %%
from pathlib import Path
import pandas as pd

# %%

CWD = Path(__file__).parent
data = CWD / "secret_du_titan"
de_file = "de.jsonl"

# open
text = pd.read_json(data / de_file, lines=True)
text.head()

# %%
