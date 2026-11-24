import pathlib, subprocess
import pandas as pd

idx = pd.read_csv("data/doclaynet_index.csv")
val = idx[idx.split == "val"]
cibles = [("perl-all-en", 1262), ("1201.4397", 190), ("NYSE_SYK_2004", 14),
          ("OTC_SDXHF_2013", 18), ("216_envr_2008", 17), ("1002.2672", 11)]

for prefixe, page in cibles:
    r = val[val.doc_name.str.startswith(prefixe) & (val.page_no == page)].iloc[0]
    txt = pathlib.Path("data/ocr_texts/val") / (pathlib.Path(r["image"]).stem + ".txt")
    ocr = txt.read_text(errors="ignore")
    print("=" * 70)
    print(r["doc_name"], "| page", page, "|", r["doc_category"])
    print("Mots de référence :", r["n_mots"], "| mots lus par l'OCR :", len(ocr.split()))
    print("Début du texte OCR :", repr(ocr.strip()[:200]))
    subprocess.run(["open", r["image"]])
