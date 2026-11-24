# Classification automatique de documents professionnels scannés

**PFA IADB 5e année · EMSI · Année universitaire 2026–2027**

Projet de comparaison entre une baseline **OCR + ML classique** (Tesseract, TF-IDF, régression logistique) et des modèles **image** (CNN) et **multimodaux** (texte + mise en page + image) pour classer le type d'un document scanné, avec une analyse détaillée des erreurs.

> Statut : projet en cours (baseline terminée, modèles image et multimodal à venir). Dernière mise à jour : 24 novembre 2026.

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
| Test de l'OCR (Tesseract) sur des échantillons | fait |
| OCR de toutes les pages (`ocr_tous.py`) | fait |
| Baseline TF-IDF + régression logistique (évaluée sur val) | fait |
| Validation croisée groupée par document et intervalle de confiance | fait |
| Analyse des erreurs de la baseline | en cours |
| Modèle image (CNN) | à faire |
| Modèle multimodal | à faire |
| Comparaison des modèles avec test apparié | à faire |
| Évaluation finale sur le jeu de test (une seule fois) | à faire |
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

- Environ 95 % des documents sont en anglais (le reste en allemand, français et japonais, selon les auteurs). Quelques pages sont en chinois.
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
├── test_ocr.py             # test d'OCR sur une page par catégorie (val)
├── ocr_echantillon.py      # OCR sur 30 pages par catégorie (train)
├── ocr_tous.py             # OCR parallèle de toutes les pages
├── baseline_tfidf.py       # baseline TF-IDF + régression logistique (entraînement train, évaluation val)
├── baseline_cv.py          # validation croisée groupée par document + IC bootstrap par document
├── analyse_erreurs.py      # erreurs par document, pages faussement classées « brevets »
├── voir_erreurs.py         # ouvre des pages mal classées et affiche leur texte OCR
├── ocr_vs_erreurs.py       # lien entre qualité de l'OCR et taux d'erreur
└── data/                   # non versionné, sauf les deux CSV de résultats
    ├── base_dataset/       # images, annotations et pdfs, par split
    ├── doclaynet_index.csv
    ├── ocr_echantillon.csv
    ├── ocr_samples/
    ├── ocr_texts/          # un fichier .txt par page, par split
    ├── baseline_val_predictions.csv
    └── baseline_val_ocr_quality.csv
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

### 5.3 Enchaînement des scripts

```bash
python explore_dataset.py          # crée data/doclaynet_index.csv, vérifie la fuite entre splits
python compte_mots.py              # ajoute la colonne n_mots (texte de référence)
python test_ocr.py                 # 1 page par catégorie, sur val (optionnel)
python ocr_echantillon.py          # 30 pages par catégorie, sur train (optionnel)
caffeinate -i python ocr_tous.py   # OCR de toutes les pages, 4 processus en parallèle
python baseline_tfidf.py           # baseline ; écrit data/baseline_val_predictions.csv
python baseline_cv.py              # validation croisée + intervalle de confiance
python analyse_erreurs.py          # analyse des erreurs par document
python ocr_vs_erreurs.py           # erreurs selon la qualité de l'OCR
```

`ocr_tous.py` écrit un fichier texte par page dans `data/ocr_texts/<split>/` et reprend là où il s'est arrêté s'il est interrompu. Contrôle de fin de traitement : `find data/ocr_texts -name "*.txt" | wc -l` doit afficher 8057. Seule la langue anglaise (`lang="eng"`) est utilisée pour l'instant.

## 6. Protocole expérimental

- **Splits officiels** de DocLayNet-base (train, val, test), conservés tels quels. Aucun document n'est partagé entre deux splits (voir 7.1).
- **Le jeu de test n'a pas été utilisé** pour entraîner ni pour choisir quoi que ce soit. L'OCR (un prétraitement sans apprentissage) lui a été appliqué, comme aux autres splits. Il sera évalué une seule fois, à la fin.
- **Baseline** : TF-IDF (unigrammes et bigrammes, `min_df=2`, 100 000 caractéristiques au maximum, `sublinear_tf`) puis régression logistique avec `class_weight="balanced"`. Le vocabulaire est appris sur le train uniquement. Les hyperparamètres sont fixés a priori, sans réglage.
- **Métrique principale : macro-F1**, car le jeu est déséquilibré (de 364 à 2 284 pages d'entraînement par classe). L'accuracy est donnée à titre indicatif.
- **Incertitude** : validation croisée en 5 plis sur le train, où les pages d'un même document restent ensemble (`StratifiedGroupKFold`), et intervalle de confiance à 95 % sur val par rééchantillonnage de documents (bootstrap, 2 000 tirages).

## 7. Résultats

Les résultats de 7.2 à 7.5 portent sur le jeu de validation (102 documents) et restent des estimations approximatives, à confirmer par la suite.

### 7.1 Absence de fuite entre splits

La comparaison des noms de fichiers d'origine (`original_filename`) montre **0 document commun** entre train, val et test : les pages d'un même PDF ne sont jamais réparties entre plusieurs splits. Cette vérification détecte les doublons exacts de nom, pas un même document republié sous un autre nom.

### 7.2 Baseline TF-IDF + régression logistique

| Mesure (val, 648 pages) | Valeur |
|---|---|
| Accuracy | 0,932 |
| **Macro-F1** | **0,917** |
| IC 95 % du macro-F1 (bootstrap par document) | [0,868 ; 0,948] |
| IC 95 % de l'accuracy | [0,889 ; 0,959] |
| Validation croisée groupée (5 plis, train), macro-F1 | 0,913 ± 0,032 |

Valeurs par pli de la validation croisée (macro-F1) : 0,926 ; 0,927 ; 0,858 ; 0,953 ; 0,900. Le pli le plus faible (0,858) n'est pas expliqué à ce stade.

| Classe | Précision | Rappel | F1 | Pages |
|---|---|---|---|---|
| manuals | 0,96 | 0,98 | 0,97 | 180 |
| scientific_articles | 0,94 | 0,95 | 0,94 | 98 |
| financial_reports | 0,95 | 0,91 | 0,93 | 171 |
| laws_and_regulations | 0,96 | 0,91 | 0,93 | 117 |
| government_tenders | 1,00 | 0,82 | 0,90 | 28 |
| patents | 0,73 | 0,94 | 0,82 | 54 |

Matrice de confusion (lignes = vraie classe, colonnes = classe prédite) :

| | financial | tenders | laws | manuals | patents | scientific |
|---|---|---|---|---|---|---|
| financial_reports | 155 | 0 | 1 | 5 | 10 | 0 |
| government_tenders | 2 | 23 | 3 | 0 | 0 | 0 |
| laws_and_regulations | 6 | 0 | 106 | 1 | 1 | 3 |
| manuals | 0 | 0 | 0 | 176 | 4 | 0 |
| patents | 0 | 0 | 0 | 0 | 51 | 3 |
| scientific_articles | 0 | 0 | 0 | 1 | 4 | 93 |

**Conséquence pour les objectifs du projet.** La cible initiale du cahier des charges (macro-F1 ≥ 0,85) est déjà atteinte par la baseline. Proposition (à valider avec l'encadrant) : mesurer le **gain** des modèles image et multimodal par rapport à la baseline, avec un bootstrap apparié par document, en particulier sur les pages où l'OCR est de mauvaise qualité.

### 7.3 Où se concentrent les erreurs

44 erreurs sur 648 pages.

- **Par document** : 19 documents sur 102 contiennent au moins une erreur, et les 5 documents les plus touchés concentrent 21 erreurs sur 44. Le score dépend donc fortement de quelques documents, ce qui justifie l'intervalle de confiance par document.
- **Classe « brevets »** : 19 pages sont faussement prédites « brevets » (10 rapports financiers, 4 manuels, 4 articles scientifiques, 1 texte de loi), d'où une précision de 0,73 alors que le rappel est de 0,94. La médiane de leurs mots de référence est de 69, et 9 de ces 19 pages ont moins de 50 mots. Pour les rapports financiers, ce sont presque toutes des pages quasi sans texte. Pour les manuels et les articles scientifiques, ce sont des pages riches en texte, concentrées sur deux documents (un manuel et un article) et probablement des pages de code ou de formules : cette dernière interprétation n'est pas vérifiée.
- **Pages quasi sans texte** (moins de 50 mots de référence) : accuracy de 0,622 sur 37 pages, contre 0,951 sur 611 pages avec 50 mots ou plus. Elles regroupent 14 des 44 erreurs.

### 7.4 Lien entre qualité de l'OCR et erreurs

Le rappel de l'OCR d'une page est la part des mots de référence retrouvés (comparaison par ensemble de mots, sans tenir compte de l'ordre).

| Rappel de l'OCR sur la page | Pages | Erreurs | Taux d'erreur |
|---|---|---|---|
| référence vide | 4 | 3 | 75,0 % |
| moins de 0,5 | 78 | 14 | 17,9 % |
| de 0,5 à 0,8 | 102 | 6 | 5,9 % |
| 0,8 ou plus | 464 | 21 | 4,5 % |

En regroupant les pages à OCR mauvais (rappel inférieur à 0,5 ou référence vide), le taux d'erreur est d'environ 20,7 % (17 erreurs sur 82 pages), contre 4,8 % (27 sur 566) pour les autres. **Mais 21 des 44 erreurs (48 %) surviennent sur des pages bien lues** : l'OCR n'explique qu'une partie des erreurs. L'autre partie vient probablement de catégories proches entre elles (par exemple 6 textes de loi pris pour des rapports financiers), à étudier.

Ce tableau montre une association, pas une causalité : les pages de formules, de tableaux ou de photos sont à la fois mal lues par l'OCR et plus difficiles à classer.

### 7.5 Types de pages où l'OCR échoue (exemples examinés)

| Type de page | Observation |
|---|---|
| Photo pleine page avec citation en texte blanc | 3 mots lus sur 12 |
| Titre clair sur fond coloré, noms en très petits caractères | 26 mots lus sur 131, le titre est manquant |
| Page entièrement en chinois | l'OCR réglé sur l'anglais produit du charabia latin |
| Tableau de numéros de versions et de dates (manuel) | chiffres confondus (`5.000£`, `5.0069`) |
| Formules mathématiques (articles scientifiques) | symboles transformés en bruit |
| Page de brevet faite de formules chimiques | l'en-tête est lu, le reste est du bruit (`A1` lu `Al`, `-continued` lu `~continue d`) |

Seules trois de ces pages ont été regardées en image (photo, titre sur fond coloré, chinois) ; pour les pages de tableau et de formules, l'analyse s'appuie seulement sur le texte OCR obtenu. Les pages comportant des portraits de personnes ne sont pas reproduites dans ce dépôt.

### 7.6 Quantité de texte par catégorie et qualité de l'OCR (exploration initiale)

Quantité de texte de référence sur les 8 057 pages (aucun OCR) :

| Catégorie | Pages | Médiane de mots | Pages avec moins de 50 mots |
|---|---|---|---|
| patents | 647 | 251 | 26,0 % |
| financial_reports | 2 627 | 483 | 6,8 % |
| laws_and_regulations | 1 287 | 387 | 5,4 % |
| government_tenders | 421 | 366 | 4,0 % |
| manuals | 1 679 | 330 | 3,9 % |
| scientific_articles | 1 396 | 426,5 | 2,3 % |

Rappel et précision médians de Tesseract sur un échantillon de 30 pages par catégorie tirées dans le train (180 pages) :

| Catégorie | Rappel médian | Précision médiane |
|---|---|---|
| manuals | 0,96 | 0,96 |
| financial_reports | 0,95 | 0,97 |
| government_tenders | 0,95 | 0,97 |
| laws_and_regulations | 0,92 | 0,97 |
| scientific_articles | 0,85 | 0,93 |
| patents | 0,63 | 0,85 |

Le rappel médian est de 0,09 sur les pages de moins de 50 mots (13 pages seulement) et de 0,93 sur les autres (167 pages). Le temps d'OCR observé est de l'ordre de 0,2 à 0,9 s par page.

## 8. Limites et précautions méthodologiques

- Le « texte de référence » est la couche texte des PDF : il peut contenir du texte invisible ou intégré à des graphiques. Un rappel faible signifie que l'OCR n'a pas retrouvé ces mots, pas forcément qu'il s'est trompé.
- Le compteur `n_mots` repose sur l'expression régulière `\w+`, qui ne segmente pas correctement le chinois ou le japonais : pour ces pages, le nombre de « mots » n'est pas fiable.
- Le critère « moins de 50 mots » est trop simple : des pages riches en texte (formules, tableaux, chinois) sont aussi mal lues. La qualité de l'OCR est un meilleur indicateur que la quantité de texte.
- Seulement 13 pages de l'échantillon d'exploration et 4 pages de val ont une référence vide ou quasi vide : les valeurs correspondantes (0,09 ; 75 %) sont des ordres de grandeur.
- Pour le macro-F1 du bootstrap, seules les classes présentes dans chaque tirage sont comptées : c'est une approximation.
- Les statistiques descriptives de 7.6 portent sur toutes les pages, test compris. Aucune décision de modélisation ne s'appuie sur le test.
- Un gain de quelques points de macro-F1 sur la baseline ne sera probablement pas significatif sans comparaison appariée sur les mêmes documents : l'écart-type entre plis est d'environ 3 points.

## 9. Prochaines étapes

1. Modèle image : ResNet-18 pré-entraîné, images réduites à 224 × 224 en niveaux de gris, fine-tuning sur le train. Hypothèse à tester : il corrige une partie des erreurs liées à un OCR de mauvaise qualité, sans rien changer aux 21 erreurs sur pages bien lues.
2. Modèle multimodal (texte, positions des mots et image), dans les mêmes conditions et avec plusieurs graines.
3. Comparaison des modèles avec un bootstrap apparié par document, globalement et par tranche de qualité d'OCR.
4. Analyse des erreurs sur les pages bien lues (catégories proches : textes de loi et rapports financiers, appels d'offres).
5. Test d'un OCR multilingue (par exemple avec le chinois) pour mesurer l'effet de la langue.
6. Évaluation finale sur le jeu de test, une seule fois, puis démonstrateur et rapport.
7. Éventuel second corpus : sous-ensemble « documents d'affaires » de RVL-CDIP.

## 10. Références

- B. Pfitzmann, C. Auer, M. Dolfi, A. S. Nassar, P. W. J. Staar. *DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation.* KDD '22. https://doi.org/10.1145/3534678.3539043
- Version réduite utilisée : https://huggingface.co/datasets/pierreguillou/DocLayNet-base
- Tesseract OCR : https://github.com/tesseract-ocr/tesseract

## 11. Auteurs

Binôme : [Mohamed Ali Azzouzi] et [Ourzhal Ghali] · Encadrant : [Younes Nadir] · EMSI, filière IADB.