import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# ============ TITRE ============
cells.append(nbf.v4.new_markdown_cell("""\
# Système d'aide à la décision pour l'agriculture togolaise
## Prototype méthodologique : recommandation de variété de culture (Maïs / Manioc / Soja) à partir de données agronomiques

**Auteur :** Komi Isaac Junior Hounbo
**Contexte :** Prototype développé en préparation du projet de recherche *"Intelligent
Decision Support for Togolese Smallholder Farmers: A Machine Learning Approach to
Agricultural Decision Support in Low-Resource Contexts"* (UFG/PPGCC).

---

### Objectif
Construire et évaluer un pipeline complet de machine learning permettant de recommander
la **culture la mieux adaptée** (Maïs, Manioc ou Soja) pour une parcelle donnée, à
partir de variables agronomiques typiques :
- Propriétés du sol (pH, azote, phosphore, texture/argile)
- Variables climatiques (précipitations cumulées, température moyenne)
- Indice de stress hydrique
- Indice de fertilité dérivé (feature engineering)

Ce cas d'usage correspond directement au **Use Case 1** (Phase 2) du projet de
recherche : recommandation de variété de culture via Random Forest / XGBoost sur
données tabulaires combinant profils de sol, prévisions de pluie et données de
rendement historiques.

### Note méthodologique sur les données
Le jeu de données utilisé ici est **synthétique**, généré pour reproduire la structure
statistique réaliste de variables agronomiques (plages de pH, azote, phosphore,
précipitations typiques des régions Maritime et Plateaux du Togo). Il permet de
**valider le pipeline méthodologique** (prétraitement, entraînement, évaluation,
interprétation) avant son application aux **données réelles ITRA** (profils de sol),
**Google Earth Engine** (NDVI/Sentinel-2/MODIS) et de la **Direction Nationale de la
Météorologie du Togo**, qui seront collectées durant la Phase 1 du projet de Master.
"""))

# ============ IMPORTS ============
cells.append(nbf.v4.new_code_cell("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)
"""))

# ============ 1. CHARGEMENT ============
cells.append(nbf.v4.new_markdown_cell("## 1. Chargement et aperçu des données"))
cells.append(nbf.v4.new_code_cell("""\
df = pd.read_csv("dataset_recommandation_culture.csv")
print("Dimensions :", df.shape)
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""\
df.describe().T
"""))

cells.append(nbf.v4.new_code_cell("""\
print("Valeurs manquantes par colonne :")
print(df.isnull().sum())
print("\\nRépartition des classes :")
print(df["culture_recommandee"].value_counts())
"""))

# ============ 2. EDA ============
cells.append(nbf.v4.new_markdown_cell("""\
## 2. Analyse exploratoire (EDA)

On examine la distribution des variables agronomiques selon la culture recommandée,
afin d'évaluer leur capacité discriminante entre les trois classes.
"""))

cells.append(nbf.v4.new_code_cell("""\
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
features_to_plot = ["pH_sol", "azote_ppm", "phosphore_ppm", "argile_pct",
                     "pluie_cumulee_mm", "temp_moy_C", "stress_hydrique", "indice_fertilite"]

for ax, col in zip(axes.flat, features_to_plot):
    sns.boxplot(data=df, x="culture_recommandee", y=col, ax=ax,
                palette=["#1b9e77", "#d95f02", "#7570b3"])
    ax.set_title(col)
    ax.set_xlabel("")

plt.tight_layout()
plt.savefig("fig_distributions.png", dpi=120)
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""\
**Observation :** Le **Soja** est associé à des sols plus riches en azote et phosphore
et à un pH plus proche de la neutralité, tandis que le **Manioc** tolère des sols plus
pauvres et acides avec moins de précipitations. Le **Maïs** se situe dans une zone
intermédiaire. L'`indice_fertilite` (variable dérivée) reflète bien cette hiérarchie,
ce qui suggère qu'il pourrait être une variable discriminante importante pour le modèle.
"""))

cells.append(nbf.v4.new_code_cell("""\
# Matrice de corrélation (variables numériques)
plt.figure(figsize=(9, 7))
corr = df.drop(columns="culture_recommandee").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation des variables agronomiques")
plt.tight_layout()
plt.savefig("fig_correlation.png", dpi=120)
plt.show()
"""))

# ============ 3. PREPROCESSING ============
cells.append(nbf.v4.new_markdown_cell("""\
## 3. Prétraitement (encodage & split train/test)

La variable cible (`culture_recommandee`) est catégorielle (3 classes) : on l'encode
numériquement. Les variables explicatives sont déjà numériques (issues du profil
agronomique de la parcelle). On effectue un split stratifié train/test (80/20) pour
préserver l'équilibre des trois classes.
"""))

cells.append(nbf.v4.new_code_cell("""\
le = LabelEncoder()
df["culture_encodee"] = le.fit_transform(df["culture_recommandee"])

print("Correspondance classes :")
for i, c in enumerate(le.classes_):
    print(f"  {i} -> {c}")

X = df.drop(columns=["culture_recommandee", "culture_encodee"])
y = df["culture_encodee"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\\nTrain :", X_train.shape, "Test :", X_test.shape)
"""))

cells.append(nbf.v4.new_code_cell("""\
# Standardisation (utile pour comparer / pour d'autres modèles type SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""))

# ============ 4. MODELE ============
cells.append(nbf.v4.new_markdown_cell("""\
## 4. Modélisation : Random Forest avec recherche d'hyperparamètres

Conformément à la méthodologie décrite (Phase 2), on entraîne un **Random Forest**
pour la recommandation de culture, avec une recherche d'hyperparamètres par
**GridSearchCV** (validation croisée 5-fold), comme prévu dans le protocole du projet.
"""))

cells.append(nbf.v4.new_code_cell("""\
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [6, 10, None],
    "min_samples_leaf": [1, 2, 4],
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Meilleurs hyperparamètres :", grid.best_params_)
print(f"Meilleur score F1 (CV) : {grid.best_score_:.3f}")

best_rf = grid.best_estimator_
"""))

# ============ 5. EVALUATION ============
cells.append(nbf.v4.new_markdown_cell("## 5. Évaluation du modèle"))

cells.append(nbf.v4.new_code_cell("""\
y_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

print(f"Accuracy : {acc:.3f}")
print(f"F1-score (weighted) : {f1:.3f}")
print()
print(classification_report(y_test, y_pred, target_names=le.classes_))
"""))

cells.append(nbf.v4.new_code_cell("""\
# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Prédiction")
plt.ylabel("Réalité")
plt.title("Matrice de confusion - Recommandation de culture")
plt.tight_layout()
plt.savefig("fig_confusion_matrix.png", dpi=120)
plt.show()
"""))

# ============ 6. FEATURE IMPORTANCE ============
cells.append(nbf.v4.new_markdown_cell("""\
## 6. Importance des variables

On analyse quelles variables agronomiques contribuent le plus à la décision du
modèle — information utile pour orienter la collecte de données prioritaires
(Phase 1 : enquêtes ITRA, profils de sol) dans le cadre du projet de Master.
"""))

cells.append(nbf.v4.new_code_cell("""\
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("Importance des variables (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=120)
plt.show()

importances
"""))

cells.append(nbf.v4.new_markdown_cell("""\
**Interprétation :** Les variables liées à la **fertilité du sol** (azote, phosphore,
indice de fertilité dérivé) et au **régime hydrique** (précipitations cumulées, stress
hydrique) apparaissent comme les plus discriminantes. Cela confirme l'importance, pour
la Phase 1 du projet, de la collecte rigoureuse des **profils de sol ITRA** et des
**séries pluviométriques de la Direction Nationale de la Météorologie**, qui constituent
les sources de données les plus déterminantes pour la qualité du modèle final.
"""))

# ============ 7. CONCLUSION ============
cells.append(nbf.v4.new_markdown_cell("""\
## 7. Conclusion et perspectives

Ce prototype démontre la faisabilité d'un pipeline complet de **recommandation de
variété de culture** (Maïs / Manioc / Soja) basé sur des variables agronomiques
(sol, climat, indices dérivés), avec des performances satisfaisantes (Random Forest +
recherche d'hyperparamètres par validation croisée).

### Limites
- Les données utilisées sont **synthétiques** (générées pour reproduire des
  distributions agronomiques réalistes), et non issues d'observations réelles de
  terrain.
- La généralisation aux conditions réelles des régions **Maritime et Plateaux** du
  Togo nécessitera l'intégration des **données ITRA**, des **enquêtes auprès de 150 à
  200 exploitants** et des **indices NDVI/Sentinel-2** prévus en Phase 1.

### Perspectives (projet de Master - UFG/PPGCC)
1. Remplacer les données synthétiques par les données réelles collectées en Phase 1
   (profils de sol ITRA, séries climatiques décennales, enquêtes terrain).
2. Étendre le modèle à un **système de recommandation multi-sortie** intégrant
   également le **timing de plantation** (LSTM sur séries pluviométriques) et la
   **détection de maladies** (CNN MobileNetV2 / TinyML).
3. Comparer Random Forest à **XGBoost** sur les données réelles, conformément au
   protocole de la Phase 2.
4. Évaluer les contraintes de déploiement (taille du modèle < 50 MB, latence
   d'inférence < 500 ms sur Android de milieu de gamme) pour une intégration dans
   l'application mobile **Flutter / TensorFlow Lite** prévue en Phase 3.
5. Co-concevoir l'interface de restitution avec les agriculteurs (icônes,
   recommandations codées par couleur) pour respecter les principes de conception
   adaptée à un faible niveau de littératie numérique.
"""))

nb["cells"] = cells

with open("/home/claude/agri_project/recommandation_culture.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook créé avec succès.")
