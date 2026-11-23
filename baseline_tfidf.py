import pathlib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

df = pd.read_csv("data/doclaynet_index.csv")

def lire(row):
    p = pathlib.Path("data/ocr_texts") / row["split"] / (pathlib.Path(row["image"]).stem + ".txt")
    return p.read_text(errors="ignore")

df["texte_ocr"] = df.apply(lire, axis=1)
train = df[df.split == "train"]
val = df[df.split == "val"]          # le test n'est PAS utilisé ici

vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
X_tr = vec.fit_transform(train["texte_ocr"])      # vocabulaire appris sur le train seulement
X_val = vec.transform(val["texte_ocr"])

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
clf.fit(X_tr, train["doc_category"])
pred = clf.predict(X_val)
y = val["doc_category"]

print(f"Accuracy : {accuracy_score(y, pred):.3f}")
print(f"Macro-F1 : {f1_score(y, pred, average='macro'):.3f}\n")
print(classification_report(y, pred, digits=2))

labels = sorted(y.unique())
cm = pd.DataFrame(confusion_matrix(y, pred, labels=labels), index=labels, columns=labels)
print("Matrice de confusion (lignes = vrai, colonnes = prédit) :")
print(cm.to_string(), "\n")

pauvre = val["n_mots"] < 50
for nom, masque in [("moins de 50 mots", pauvre), ("50 mots ou plus", ~pauvre)]:
    if masque.sum():
        acc = accuracy_score(y[masque], pred[masque])
        print(f"Accuracy, pages avec {nom} : {acc:.3f} ({masque.sum()} pages)")

out = val[["image", "doc_category", "n_mots"]].copy()
out["prediction"] = pred
out.to_csv("data/baseline_val_predictions.csv", index=False)
print("\nPrédictions sauvegardées : data/baseline_val_predictions.csv")
