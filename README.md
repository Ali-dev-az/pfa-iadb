# Classification automatique de documents professionnels scannés

**PFA IADB 5e année · EMSI · Année universitaire 2026–2027**

Projet de comparaison entre une baseline **OCR + ML classique** (Tesseract, TF-IDF, régression logistique) et des modèles **image** (CNN) et **multimodaux** (texte + mise en page + image) pour classer le type d'un document scanné, avec une analyse détaillée des erreurs.

> Statut : projet en cours (phase données et prétraitement). Dernière mise à jour : 17 novembre 2026.

---

## 1. Problématique

Les organisations qui traitent des dossiers (leasing, crédit, administration) reçoivent de nombreux documents scannés dont il faut d'abord identifier le type avant de les orienter. Ce projet étudie dans quelle mesure cette étape peut être automatisée, et si un modèle qui exploite la mise en page et l'image apporte un gain mesurable par rapport à une approche textuelle simple.

## 2. État d'avancement

| Étape | État |
|---|---|
| Choix du sujet et cahier des charges | fait |
| Téléchargement du jeu de données DocLayNet-base | fait |
| Indexation des 8 057 pages et vérification de l'absence de fuite entre splits | fait |
| Analyse exploratoire : quantité de texte par catégorie | fait |
| Test de l'OCR (Tesseract) sur un échantillon | fait |
| OCR de toutes les pages (`ocr_tous.py`) | **en cours** |
| Baseline TF-IDF + régression logistique | à faire |
| Modèle image (CNN) | à faire |
| Modèle multimodal | à faire |
| Analyse des erreurs | à faire |
| Démonstrateur (Streamlit ou Gradio) | à faire |

## 3. Données

**Jeu utilisé : [DocLayNet](https://github.com/DS4SD/DocLayNet) (IBM Research), version réduite « base »** (`pierreguillou/DocLayNet-base` sur Hugging Face, environ 10 % du jeu complet).

- Licence indiquée par les auteurs : CDLA-Permissive-1.0.
- 6 catégories de documents (`doc_category`) : `financial_reports`, `government_tenders`, `laws_and_regulations`, `manuals`, `patents`, `scientific_articles`.
- Pour chaque page : une image PNG, un PDF d'une page et un fichier JSON d'annotations (métadonnées, texte de référence avec positions).

| Catégorie | Train | Val | Test |
|---|---|---|---|
| financial_reports | 2 284 | 171 | 172 |
| manuals | 1 422 | 180 | 77 |
| scientific_articles | 1 195 | 98 | 103 |
| laws_and_regulations | 1 084 | 117 | 86 |
| patents | 561 | 54 | 32 |
| government_tenders | 364 | 28 | 29 |
| **Total (pages)** | **6 910** | **648** | **499** |
| Documents d'origine distincts | 1 590 | 102 | 116 |

**Les données ne sont pas versionnées dans ce dépôt** (plusieurs Go). Voir la section « Reproduire » pour les récupérer.

### Limites connues du jeu

- Environ 95 % des documents sont en anglais (le reste en allemand, français et japonais, selon les auteurs).
- Les images sont redimensionnées en carré de 1025 × 1025 pixels : les proportions d'origine sont perdues (les dimensions d'origine sont conservées dans les métadonnées).
- Les 6 catégories sont grossières par rapport à d'autres jeux de classification de documents.
- Plusieurs pages proviennent du même document : elles ne sont pas indépendantes. La taille d'échantillon réelle du test est d'environ 116 documents, pas 499.
- Très petit effectif de test pour `government_tenders` (29 pages) et `patents` (32 pages).
- Aucun document marocain réel : ils relèvent de données personnelles et ne sont pas utilisés.

## 4. Structure du dépôt

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── explore_dataset.py      # indexe les pages, contrôle la fuite entre splits
├── compte_mots.py          # ajoute le nombre de mots de référence par page
├── test_ocr.py             # test d'OCR sur une page par catégorie (validation)
├── ocr_echantillon.py      # OCR sur 30 pages par catégorie (train)
├── ocr_tous.py             # OCR parallèle de toutes les pages
└── data/                   # non versionné, sauf les deux CSV de résultats
    ├── base_dataset/       # images, annotations et pdfs, par split
    ├── doclaynet_index.csv
    ├── ocr_echantillon.csv
    ├── ocr_samples/
    └── ocr_texts/          # un fichier .txt par page, par split
```

## 5. Reproduire

Testé sur macOS, Python 3.13.

### 5.1 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install tesseract
tesseract --version
```

### 5.2 Récupérer les données

Les versions récentes de la bibliothèque `datasets` (4.x) ne gèrent plus les jeux de données à script, ce qui est le cas de `DocLayNet-base`. On utilise donc `datasets<4` (voir `requirements.txt`).

```bash
python -c "
from datasets import load_dataset
load_dataset('pierreguillou/DocLayNet-base', trust_remote_code=True)
"
```

Cette commande télécharge l'archive (environ 3,8 Go) puis **échoue lors de la conversion** avec l'erreur `Float value ... was truncated converting to int64`. C'est attendu : le script du jeu déclare des coordonnées entières alors que certaines sont décimales. L'archive reste dans le cache Hugging Face, et nous la lisons directement sans passer par ce chargeur :

```bash
ZIP=$(find ~/.cache/huggingface -name "dataset_base.zip" | head -1)
mkdir -p data
unzip -q "$ZIP" -d data/
```

Il faut environ 4,5 Go d'espace libre. On obtient `data/base_dataset/{train,val,test}/{images,annotations,pdfs}` avec respectivement 6 910, 648 et 499 fichiers par dossier.

### 5.3 Indexation et exploration

```bash
python explore_dataset.py     # crée data/doclaynet_index.csv, vérifie la fuite entre splits
python compte_mots.py         # ajoute la colonne n_mots (texte de référence)
```

### 5.4 OCR

```bash
python test_ocr.py            # 1 page par catégorie, sur val
python ocr_echantillon.py     # 30 pages par catégorie, sur train
caffeinate -i python ocr_tous.py   # toutes les pages, 4 processus en parallèle
```

`ocr_tous.py` écrit un fichier texte par page dans `data/ocr_texts/<split>/` et **reprend là où il s'est arrêté** s'il est interrompu (les pages déjà traitées sont ignorées). Contrôle de fin de traitement :

```bash
find data/ocr_texts -name "*.txt" | wc -l     # doit afficher 8057
```

Seule la langue anglaise (`lang="eng"`) est utilisée pour l'instant.

## 6. Résultats préliminaires

Résultats exploratoires, obtenus sur de petits échantillons : ils indiquent des tendances, pas des performances finales.

### 6.1 Absence de fuite entre splits

La comparaison des noms de fichiers d'origine (`original_filename`) montre **0 document commun** entre train, val et test : les pages d'un même PDF ne sont jamais réparties entre plusieurs splits. Cette vérification détecte les doublons exacts de nom, pas un même document republié sous un autre nom.

### 6.2 Quantité de texte par catégorie

Calculée sur le texte de référence des 8 057 pages (aucun OCR) :

| Catégorie | Pages | Médiane de mots | Pages avec moins de 50 mots |
|---|---|---|---|
| patents | 647 | 251 | 26,0 % |
| financial_reports | 2 627 | 483 | 6,8 % |
| laws_and_regulations | 1 287 | 387 | 5,4 % |
| government_tenders | 421 | 366 | 4,0 % |
| manuals | 1 679 | 330 | 3,9 % |
| scientific_articles | 1 396 | 426,5 | 2,3 % |

Les brevets contiennent beaucoup plus de pages quasi sans texte (dessins de structures chimiques, schémas) que les autres catégories.

### 6.3 Qualité de l'OCR (Tesseract)

Échantillon de 30 pages par catégorie tirées dans le train (180 pages). Le rappel est la part des mots de référence retrouvés par l'OCR (comparaison par ensemble de mots, sans tenir compte de l'ordre) ; la précision est la part des mots lus qui existent dans la référence.

| Catégorie | Rappel médian | Précision médiane |
|---|---|---|
| manuals | 0,96 | 0,96 |
| financial_reports | 0,95 | 0,97 |
| government_tenders | 0,95 | 0,97 |
| laws_and_regulations | 0,92 | 0,97 |
| scientific_articles | 0,85 | 0,93 |
| patents | 0,63 | 0,85 |

| Quantité de texte sur la page | Rappel médian | Nombre de pages |
|---|---|---|
| moins de 50 mots | 0,09 | 13 |
| 50 mots ou plus | 0,93 | 167 |

L'OCR est excellent sur les pages riches en texte et ne retrouve presque rien sur les pages qui en contiennent peu. Le temps d'OCR observé est de l'ordre de 0,2 à 0,9 s par page.

### 6.4 Exemple d'erreurs OCR (brevet)

Sur une page de brevet constituée de formules chimiques, Tesseract lit correctement l'en-tête mais produit du bruit sur les dessins. Exemples : `A1` lu `Al`, `-continued` lu `~continue d`, et des suites de caractères parasites sur les structures.

## 7. Limites et précautions méthodologiques

- Le « texte de référence » est la couche texte des PDF : il peut contenir du texte invisible ou intégré à des graphiques. Un rappel faible signifie que l'OCR n'a pas retrouvé ces mots, pas forcément qu'il s'est trompé.
- Seulement 13 pages de l'échantillon ont moins de 50 mots : la valeur 0,09 est un ordre de grandeur.
- Ces mesures portent sur la qualité de l'OCR, pas encore sur la classification. L'effet du bruit OCR sur la classification sera mesuré avec la baseline.
- Les statistiques descriptives de la section 6.2 portent sur toutes les pages, test compris. Aucune décision de modélisation ne s'appuie sur le test. Les tests d'OCR exploratoires utilisent val (test d'une page par catégorie) et train (échantillon de 180 pages).
- Pour la sélection des hyperparamètres : validation croisée groupée par document (`GroupKFold`) sur le train, et rééchantillonnage par document pour les intervalles de confiance.
- Le jeu étant déséquilibré, la métrique principale est la **macro-F1**.

## 8. Prochaines étapes

1. Terminer l'OCR sur toutes les pages et vérifier le compte (8 057 fichiers).
2. Baseline : TF-IDF + régression logistique, avec validation groupée par document et poids de classe ; mesurer macro-F1, matrice de confusion, précision et rappel par classe.
3. Comparer à un CNN pré-entraîné (fine-tuning) puis à un modèle multimodal, dans les mêmes conditions, avec plusieurs graines.
4. Analyse des erreurs : classes confondues, effet des pages pauvres en texte, effet du bruit OCR (comparaison OCR contre texte de référence).
5. Éventuel second corpus : sous-ensemble « documents d'affaires » de RVL-CDIP.
6. Démonstrateur et rapport.

## 9. Références

- B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, P. W. J. Staar. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD '22. https://doi.org/10.1145/3534678.3539043
- Version réduite utilisée : https://huggingface.co/datasets/pierreguillou/DocLayNet-base
- Tesseract OCR : https://github.com/tesseract-ocr/tesseract

## 10. Auteurs

Binôme : [Med Ali Azzouzi] et [Ourhzal Ghali] · Encadrant : [Younes Nadir] · EMSI, filière IADB.# pfa-iadb
