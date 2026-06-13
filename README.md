# Système de recommandation de culture pour l'agriculture togolaise (prototype méthodologique)

Prototype de machine learning développé en préparation du projet de recherche de
Master **"Intelligent Decision Support for Togolese Smallholder Farmers: A Machine
Learning Approach to Agricultural Decision Support in Low-Resource Contexts"**,

**Auteur :** Komi Isaac Junior Hounbo

## Objectif

Construire un pipeline complet de machine learning permettant de recommander la
culture la mieux adaptée (**Maïs, Manioc ou Soja**) à une parcelle donnée, à partir de
variables agronomiques : propriétés du sol (pH, azote, phosphore, texture), variables
climatiques (précipitations cumulées, température moyenne) et stress hydrique.

Ce cas d'usage correspond au **Use Case 1** (Phase 2 — Modélisation) du projet de
recherche : recommandation de variété de culture via Random Forest / XGBoost sur
données tabulaires combinant profils de sol, prévisions de pluie et historique de
rendement.

## Données

Le jeu de données utilisé (`dataset_recommandation_culture.csv`) est **synthétique**,
généré pour reproduire la structure statistique réaliste de variables agronomiques
typiques des régions Maritime et Plateaux du Togo. Il permet de **valider le pipeline
méthodologique** avant son application aux données réelles qui seront collectées en
Phase 1 du projet de Master :
- Enquêtes auprès de 150–200 exploitants (ITRA)
- Profils de sol (base agronomique ITRA)
- Imagerie satellite Sentinel-2 / MODIS (Google Earth Engine) — indices NDVI
- Séries climatiques décennales (Direction Nationale de la Météorologie du Togo)

## Méthodologie

1. Analyse exploratoire (distribution des variables par culture, matrice de
   corrélation)
2. Feature engineering : indice de fertilité composite (azote, phosphore, pH)
3. Encodage de la variable cible et split stratifié train/test (80/20)
4. Modélisation : Random Forest avec recherche d'hyperparamètres (GridSearchCV,
   validation croisée 5-fold)
5. Évaluation : accuracy, F1-score pondéré, matrice de confusion
6. Interprétation : importance des variables (feature importance)

## Résultats

- **Accuracy :** 0.785
- **F1-score pondéré :** 0.784

Les variables liées à la fertilité du sol (indice de fertilité, azote, phosphore) et
au régime hydrique (précipitations, stress hydrique) sont les plus discriminantes,
confirmant l'importance de la collecte de données de sol et de climat prévue en
Phase 1 du projet.

## Structure du dépôt

```
.
├── 01_generate_dataset.py          # Génération du dataset synthétique
├── 02_build_notebook.py            # Génération du notebook Jupyter
├── dataset_recommandation_culture.csv
├── recommandation_culture.ipynb    # Notebook complet (EDA -> modèle -> évaluation)
├── fig_distributions.png
├── fig_correlation.png
├── fig_confusion_matrix.png
├── fig_feature_importance.png
└── README.md
```

## Perspectives

- Remplacement des données synthétiques par les données réelles ITRA / Google Earth
  Engine / Météo Togo collectées en Phase 1
- Extension vers un système multi-sortie (timing de plantation via LSTM, détection de
  maladies via CNN MobileNetV2 / TinyML)
- Comparaison Random Forest vs XGBoost sur données réelles
- Optimisation pour déploiement mobile (TensorFlow Lite, < 50 MB, < 500 ms
  d'inférence)

## Licence

Projet académique — usage libre à but de recherche et d'enseignement.
