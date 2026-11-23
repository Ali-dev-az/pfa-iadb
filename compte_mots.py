import json, pathlib, re
import pandas as pd

df = pd.read_csv("data/doclaynet_index.csv")

def nb_mots(img):
    ann = pathlib.Path(img.replace("/images/", "/annotations/")).with_suffix(".json")
    d = json.loads(ann.read_text())
    return len(re.findall(r"\w+", " ".join(i.get("text", "") for i in d["form"])))

df["n_mots"] = df["image"].apply(nb_mots)
df.to_csv("data/doclaynet_index.csv", index=False)

g = df.groupby("doc_category")["n_mots"]
print(pd.DataFrame({
    "pages": g.size(),
    "mediane_mots": g.median(),
    "pct_moins_de_50_mots": g.apply(lambda s: round((s < 50).mean() * 100, 1)),
}))
