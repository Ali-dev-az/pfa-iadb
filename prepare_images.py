import numpy as np
import pandas as pd
from PIL import Image

SIZE = 224
df = pd.read_csv("data/doclaynet_index.csv")

for split in ["train", "val", "test"]:
    d = df[df.split == split].reset_index(drop=True)
    X = np.zeros((len(d), SIZE, SIZE), dtype=np.uint8)
    for i, p in enumerate(d["image"]):
        X[i] = np.asarray(Image.open(p).convert("L").resize((SIZE, SIZE), Image.BILINEAR))
        if (i + 1) % 1000 == 0:
            print(split, i + 1, "/", len(d), flush=True)
    np.save(f"data/images_{SIZE}_{split}.npy", X)
    d[["image", "doc_category", "doc_name"]].to_csv(f"data/meta_{split}.csv", index=False)
    print(split, "terminé :", X.shape)
