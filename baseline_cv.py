import pathlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

df = pd.read_csv("data/doclaynet_index.csv")

def lire(row):
    p = pathlib.Path("data/ocr_texts") / row["split"] / (pathlib.Path(row["image"]).stem + ".txt")
    return p.read_text(errors="ignore")

df["texte_ocr"] = df.apply(lire, axis=1)
train = df[df.split == "train"].reset_index(drop=True)

def modele():
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0)
    return vec, clf

# 1) Validation croisée groupée par document (train seulement)
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
scores = []
for k, (i_tr, i_te) in enumerate(cv.split(train, train["doc_category"], groups=train["doc_name"]), 1):
    vec, clf = modele()
    clf.fit(vec.fit_transform(train.loc[i_tr, "texte_ocr"]), train.loc[i_tr, "doc_category"])
    pred = clf.predict(vec.transform(train.loc[i_te, "texte_ocr"]))
    y = train.loc[i_te, "doc_category"]
    f1, acc = f1_score(y, pred, average="macro"), accuracy_score(y, pred)
    scores.append((f1, acc))
    print(f"Pli {k} : macro-F1 = {f1:.3f} | accuracy = {acc:.3f} | documents = {train.loc[i_te, 'doc_name'].nunique()}")
s = np.array(scores)
print(f"\nValidation croisée (5 plis) : macro-F1 = {s[:,0].mean():.3f} ± {s[:,0].std():.3f}"
      f" | accuracy = {s[:,1].mean():.3f} ± {s[:,1].std():.3f}")

# 2) Intervalle de confiance sur val, bootstrap par document
p = pd.read_csv("data/baseline_val_predictions.csv").merge(df[["image", "doc_name"]], on="image")
y_true, y_pred = p["doc_category"].values, p["prediction"].values
groupes = {d: g.index.values for d, g in p.groupby("doc_name")}
docs = np.array(list(groupes))
rng = np.random.default_rng(0)
f1s, accs = [], []
for _ in range(2000):
    tires = rng.choice(docs, size=len(docs), replace=True)
    idx = np.concatenate([groupes[d] for d in tires])
    yt, yp = y_true[idx], y_pred[idx]
    f1s.append(f1_score(yt, yp, average="macro", labels=np.unique(yt), zero_division=0))
    accs.append(accuracy_score(yt, yp))

print(f"\nValidation officielle ({len(docs)} documents), estimation ponctuelle :"
      f" macro-F1 = {f1_score(y_true, y_pred, average='macro'):.3f}"
      f" | accuracy = {accuracy_score(y_true, y_pred):.3f}")
print("IC 95 %% (bootstrap par document) : macro-F1 = [%.3f ; %.3f] | accuracy = [%.3f ; %.3f]"
      % (*np.percentile(f1s, [2.5, 97.5]), *np.percentile(accs, [2.5, 97.5])))
