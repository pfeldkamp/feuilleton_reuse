# %%
from pathlib import Path
import pandas as pd

# %%

CWD = Path(__file__).parent
data = CWD / "mysteres"
de_file = "die-geheimnisse-von-paris.txt"

with open(data / de_file, "r", encoding="utf-8") as f:
    text = f.read()

print(text)
# %%
