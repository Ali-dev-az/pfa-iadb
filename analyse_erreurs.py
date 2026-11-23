import pandas as pd

p = pd.read_csv("data/baseline_val_predictions.csv")
idx = pd.read_csv("data/doclaynet_index.csv")[["image", "doc_name", "page_no"]]
p = p.merge(idx, on="image")
p["erreur"] = p["doc_category"] != p["prediction"]
n_err = int(p["erreur"].sum())
print("Erreurs :", n_err, "sur", len(p), "pages\n")

par_doc = (p.groupby("doc_name")
             .agg(pages=("erreur", "size"), erreurs=("erreur", "sum"),
                  categorie=("doc_category", "first"))
             .query("erreurs > 0")
             .sort_values("erreurs", ascending=False))
print("Documents avec au moins une erreur :", len(par_doc), "sur", p["doc_name"].nunique())
print(par_doc.head(10).to_string())
top5 = int(par_doc.head(5)["erreurs"].sum())
print(f"\nLes 5 documents les plus touchés concentrent {top5} erreurs sur {n_err}\n")

fp = p[(p["prediction"] == "patents") & (p["doc_category"] != "patents")]
print("Pages faussement étiquetées brevets :", len(fp))
print("Mots (texte de référence), médiane :", fp["n_mots"].median())
print("Parmi elles, pages avec moins de 50 mots :", int((fp["n_mots"] < 50).sum()))
print(fp[["doc_category", "doc_name", "page_no", "n_mots"]].to_string(index=False))
