import json, pathlib, re
from collections import Counter
import pandas as pd

mots = lambda t: re.findall(r"\w+", t.lower())
p = pd.read_csv("data/baseline_val_predictions.csv")
p["erreur"] = p["doc_category"] != p["prediction"]

def rappel(img):
    img = pathlib.Path(img)
    ann = pathlib.Path(str(img).replace("/images/", "/annotations/")).with_suffix(".json")
    d = json.loads(ann.read_text())
    ref = Counter(mots(" ".join(i.get("text", "") for i in d["form"])))
    ocr_txt = (pathlib.Path("data/ocr_texts/val") / (img.stem + ".txt")).read_text(errors="ignore")
    n = sum(ref.values())
    return sum((ref & Counter(mots(ocr_txt))).values()) / n if n else None

p["rappel_ocr"] = p["image"].apply(rappel)

def tranche(r):
    if pd.isna(r): return "0 - référence vide"
    if r < 0.5:    return "1 - rappel < 0,5"
    if r < 0.8:    return "2 - rappel de 0,5 à 0,8"
    return "3 - rappel >= 0,8"

p["tranche"] = p["rappel_ocr"].apply(tranche)
res = p.groupby("tranche").agg(pages=("erreur", "size"), erreurs=("erreur", "sum"))
res["taux_erreur_%"] = (100 * res["erreurs"] / res["pages"]).round(1)
print(res)
print("\nTotal :", int(p["erreur"].sum()), "erreurs sur", len(p), "pages")

p.to_csv("data/baseline_val_ocr_quality.csv", index=False)
