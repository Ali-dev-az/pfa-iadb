import json, pathlib
import pandas as pd

ROOT = pathlib.Path("data/base_dataset")

def find_key(obj, key):
    """Cherche une clé n'importe où dans un JSON imbriqué."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj[:50]:
            r = find_key(v, key)
            if r is not None:
                return r
    return None

# 1) Diagnostic : à quoi ressemble un fichier d'annotations ?
first = next((ROOT / "train" / "annotations").glob("*.json"))
data = json.loads(first.read_text())
print("Fichier :", first.name)
print("Type :", type(data).__name__)
if isinstance(data, dict):
    print("Clés principales :", list(data.keys())[:20])

# 2) Construire l'index de toutes les pages
rows = []
for split in ["train", "val", "test"]:
    for f in (ROOT / split / "annotations").glob("*.json"):
        d = json.loads(f.read_text())
        img = ROOT / split / "images" / (f.stem + ".png")
        rows.append({
            "split": split,
            "image": str(img),
            "image_existe": img.exists(),
            "doc_category": find_key(d, "doc_category"),
            "doc_name": find_key(d, "original_filename"),
            "page_no": find_key(d, "page_no"),
        })

df = pd.DataFrame(rows)
df.to_csv("data/doclaynet_index.csv", index=False)

# 3) Résumé
print("\nPages indexées :", len(df))
print("Images retrouvées :", int(df["image_existe"].sum()), "/", len(df))
print("Valeurs manquantes :", df.isna().sum().to_dict())
print("\nEffectifs par catégorie et par split :")
print(df.groupby(["doc_category", "split"], dropna=False).size().unstack("split", fill_value=0))

# 4) Fuite possible : documents d'origine communs entre splits
sets = {s: set(df[df.split == s]["doc_name"].dropna()) for s in ["train", "val", "test"]}
print("\nDocuments communs train/test :", len(sets["train"] & sets["test"]))
print("Documents communs train/val  :", len(sets["train"] & sets["val"]))
print("Documents communs val/test   :", len(sets["val"] & sets["test"]))
print("Documents distincts par split :", {s: len(v) for s, v in sets.items()})
