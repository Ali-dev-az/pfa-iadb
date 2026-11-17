import json, re, time, pathlib
from collections import Counter
import pandas as pd
import pytesseract
from PIL import Image

df = pd.read_csv("data/doclaynet_index.csv")
val = df[df.split == "val"]
pages = val.groupby("doc_category").sample(1, random_state=0)

def mots(texte):
    return re.findall(r"\w+", texte.lower())

out_dir = pathlib.Path("data/ocr_samples")
out_dir.mkdir(exist_ok=True)
resultats = []
premier = True

for _, row in pages.iterrows():
    img_path = pathlib.Path(row["image"])
    ann_path = pathlib.Path(str(img_path).replace("/images/", "/annotations/")).with_suffix(".json")
    d = json.loads(ann_path.read_text())

    if premier:
        print("Exemple d'élément 'form' :")
        print(json.dumps(d["form"][0], indent=2)[:600], "\n")
        premier = False

    ref = " ".join(item.get("text", "") for item in d["form"])
    t0 = time.time()
    ocr = pytesseract.image_to_string(Image.open(img_path), lang="eng")
    duree = time.time() - t0
    (out_dir / (img_path.stem + ".txt")).write_text(ocr)

    c_ref, c_ocr = Counter(mots(ref)), Counter(mots(ocr))
    commun = sum((c_ref & c_ocr).values())
    n_ref, n_ocr = sum(c_ref.values()), sum(c_ocr.values())
    resultats.append({
        "categorie": row["doc_category"],
        "mots_ref": n_ref,
        "mots_ocr": n_ocr,
        "rappel": round(commun / n_ref, 2) if n_ref else None,
        "precision": round(commun / n_ocr, 2) if n_ocr else None,
        "secondes": round(duree, 1),
    })

print(pd.DataFrame(resultats).to_string(index=False))
print("\nTextes OCR sauvegardés dans", out_dir)
