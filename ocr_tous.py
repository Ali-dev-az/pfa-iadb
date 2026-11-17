import os, pathlib
os.environ["OMP_THREAD_LIMIT"] = "1"   # évite que Tesseract sature le CPU
from multiprocessing import Pool
import pandas as pd
import pytesseract
from PIL import Image

OUT = pathlib.Path("data/ocr_texts")

def run(args):
    img, split = args
    p = pathlib.Path(img)
    out = OUT / split / (p.stem + ".txt")
    if out.exists():
        return 0
    try:
        txt = pytesseract.image_to_string(Image.open(p), lang="eng")
    except Exception as e:
        print("ERREUR", p.name, e, flush=True)
        return 0                         # pas de fichier écrit : une relance réessaiera
    out.write_text(txt)
    return 1

if __name__ == "__main__":
    df = pd.read_csv("data/doclaynet_index.csv")
    for s in ["train", "val", "test"]:
        (OUT / s).mkdir(parents=True, exist_ok=True)
    jobs = list(zip(df["image"], df["split"]))
    with Pool(4) as pool:
        for i, _ in enumerate(pool.imap_unordered(run, jobs, chunksize=8), 1):
            if i % 200 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    print("Terminé :", len(jobs), "pages traitées")
