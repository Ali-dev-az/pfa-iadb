import json, re, pathlib
from collections import Counter
import pandas as pd
import pytesseract
from PIL import Image

df = pd.read_csv("data/doclaynet_index.csv")
tr = df[df.split == "train"]
ech = tr.groupby("doc_category").sample(30, random_state=1)

mots = lambda t: re.findall(r"\w+", t.lower())
lignes = []
for _, r in ech.iterrows():
    ann = pathlib.Path(r["image"].replace("/images/", "/annotations/")).with_suffix(".json")
    d = json.loads(ann.read_text())
    ref = " ".join(i.get("text", "") for i in d["form"])
    ocr = pytesseract.image_to_string(Image.open(r["image"]), lang="eng")
    cr, co = Counter(mots(ref)), Counter(mots(ocr))
    commun = sum((cr & co).values())
    lignes.append({
        "categorie": r["doc_category"],
        "n_mots": r["n_mots"],
        "rappel": commun / sum(cr.values()) if cr else None,
        "precision": commun / sum(co.values()) if co else None,
    })

res = pd.DataFrame(lignes)
res.to_csv("data/ocr_echantillon.csv", index=False)
print("Rappel et précision médians par catégorie :")
print(res.groupby("categorie")[["rappel", "precision"]].median().round(2))
res["texte"] = res["n_mots"].apply(lambda n: "< 50 mots" if n < 50 else ">= 50 mots")
print("\nRappel médian selon la quantité de texte :")
print(res.groupby("texte")["rappel"].agg(["median", "count"]).round(2))
