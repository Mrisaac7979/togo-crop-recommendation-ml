"""
Génération d'un jeu de données synthétique simulant des observations
agronomiques (sol, climat, pratiques culturales) pour la classification
"variété de culture la mieux adaptée" parmi maïs, manioc et soja
(cultures cibles des régions Maritime et Plateaux, Togo).

NOTE METHODOLOGIQUE :
Ce dataset est SYNTHETIQUE mais construit pour reproduire la structure
statistique typique des variables agronomiques (pH du sol, azote,
phosphore, texture, précipitations cumulées, température moyenne,
indice de stress hydrique). Il sert de prototype méthodologique pour
valider le pipeline ML (preprocessing -> entraînement -> évaluation
-> interprétation) avant son application aux données réelles ITRA,
Google Earth Engine (NDVI/Sentinel-2/MODIS) et de la Direction
Nationale de la Météorologie du Togo, dans le cadre du projet de
Master (UFG/PPGCC).
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_CLASS = 1200  # 3 classes -> 3600 échantillons

CLASSES = ["Maïs", "Manioc", "Soja"]

def make_class(n, label, params):
    data = {}
    for col, (mu, sigma) in params.items():
        data[col] = np.random.normal(mu, sigma, n)
    df = pd.DataFrame(data)
    df["culture_recommandee"] = label
    return df

# ---- Paramètres agronomiques réalistes par culture ----
# pH_sol (0-14), azote_ppm, phosphore_ppm, argile_pct (texture),
# pluie_cumulee_mm (saison), temp_moy_C, stress_hydrique (0-1, 0=aucun)

maize_params = {
    "pH_sol":             (6.2, 0.5),
    "azote_ppm":          (45, 10),
    "phosphore_ppm":      (22, 6),
    "argile_pct":         (28, 7),
    "pluie_cumulee_mm":   (950, 150),
    "temp_moy_C":         (27.5, 1.5),
    "stress_hydrique":    (0.30, 0.12),
}

cassava_params = {  # Manioc : tolère sols pauvres, plus de chaleur, moins d'eau
    "pH_sol":             (5.6, 0.6),
    "azote_ppm":          (28, 9),
    "phosphore_ppm":      (15, 5),
    "argile_pct":         (20, 8),
    "pluie_cumulee_mm":   (800, 180),
    "temp_moy_C":         (28.5, 1.6),
    "stress_hydrique":    (0.45, 0.15),
}

soybean_params = {  # Soja : sols plus riches en azote/phosphore, pluie modérée
    "pH_sol":             (6.6, 0.5),
    "azote_ppm":          (55, 12),
    "phosphore_ppm":      (30, 7),
    "argile_pct":         (32, 6),
    "pluie_cumulee_mm":   (1050, 140),
    "temp_moy_C":         (26.8, 1.4),
    "stress_hydrique":    (0.25, 0.10),
}

df_maize = make_class(N_PER_CLASS, "Maïs", maize_params)
df_cassava = make_class(N_PER_CLASS, "Manioc", cassava_params)
df_soybean = make_class(N_PER_CLASS, "Soja", soybean_params)

df = pd.concat([df_maize, df_cassava, df_soybean], ignore_index=True)

# Contraintes physiques réalistes
df["pH_sol"] = df["pH_sol"].clip(4.0, 8.5)
df["azote_ppm"] = df["azote_ppm"].clip(lower=2)
df["phosphore_ppm"] = df["phosphore_ppm"].clip(lower=1)
df["argile_pct"] = df["argile_pct"].clip(5, 60)
df["pluie_cumulee_mm"] = df["pluie_cumulee_mm"].clip(lower=200)
df["stress_hydrique"] = df["stress_hydrique"].clip(0, 1)

# ---- Variable dérivée : indice de fertilité (feature engineering) ----
# combinaison simple azote/phosphore/pH, typique d'un indice agronomique composite
df["indice_fertilite"] = (
    0.4 * (df["azote_ppm"] / df["azote_ppm"].max())
    + 0.4 * (df["phosphore_ppm"] / df["phosphore_ppm"].max())
    + 0.2 * (1 - abs(df["pH_sol"] - 6.5) / 2.5)
)

# Mélanger les lignes
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Bruit de label (8%) : simule la variabilité réelle des décisions agronomiques
# (terrains limites, micro-zones, pratiques mixtes)
flip_idx = np.random.choice(df.index, size=int(0.08 * len(df)), replace=False)
for idx in flip_idx:
    current = df.loc[idx, "culture_recommandee"]
    choices = [c for c in CLASSES if c != current]
    df.loc[idx, "culture_recommandee"] = np.random.choice(choices)

df.to_csv("/home/claude/agri_project/dataset_recommandation_culture.csv", index=False)
print("Dataset généré :", df.shape)
print(df.head())
print("\nRépartition des classes :")
print(df["culture_recommandee"].value_counts())
