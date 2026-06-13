# Crop Recommendation System for Togolese Smallholder Farmers (Methodological Prototype)

Machine learning prototype developed in preparation for the Master's research project
**"Intelligent Decision Support for Togolese Smallholder Farmers: A Machine Learning
Approach to Agricultural Decision Support in Low-Resource Contexts"**, submitted to
the **Graduate Program in Computer Science (PPGCC)**, Universidade Federal de Goiás
(UFG), Brazil.

**Author:** Komi Isaac Junior Hounbo

## Objective

Build a complete machine learning pipeline to recommend the most suitable crop
(**Maize, Cassava, or Soybean**) for a given plot, based on agronomic variables: soil
properties (pH, nitrogen, phosphorus, clay content), climate variables (cumulative
rainfall, average temperature), and water stress index.

This use case corresponds to **Use Case 1** (Phase 2 — Modeling) of the research
project: crop variety recommendation via Random Forest / XGBoost on tabular data
combining soil profiles, rainfall forecasts, and historical yield data.

## Data

The dataset used (`dataset_recommandation_culture.csv`) is **synthetic**, generated to
reproduce the realistic statistical structure of agronomic variables typical of the
Maritime and Plateaux regions of Togo. It serves to **validate the methodological
pipeline** before applying it to the real data to be collected in Phase 1 of the
Master's project:
- Surveys of 150–200 farmers (in collaboration with ITRA)
- Soil profile data (ITRA agronomic database)
- Satellite imagery from Sentinel-2 / MODIS (Google Earth Engine) — NDVI indices
- Ten-year climate time series (Togo's National Meteorological Directorate)

## Methodology

1. Exploratory data analysis (variable distributions per crop, correlation matrix)
2. Feature engineering: composite soil fertility index (nitrogen, phosphorus, pH)
3. Target encoding and stratified train/test split (80/20)
4. Modeling: Random Forest with hyperparameter tuning (GridSearchCV, 5-fold
   cross-validation)
5. Evaluation: accuracy, weighted F1-score, confusion matrix
6. Interpretation: feature importance analysis

## Results

- **Accuracy:** 0.785
- **Weighted F1-score:** 0.784

Variables related to soil fertility (fertility index, nitrogen, phosphorus) and water
regime (rainfall, water stress) are the most discriminative, confirming the
importance of the soil and climate data collection planned for Phase 1 of the
project.

## Repository structure

```
.
├── 01_generate_dataset.py          # Synthetic dataset generation
├── 02_build_notebook.py            # Jupyter notebook generation
├── dataset_recommandation_culture.csv
├── recommandation_culture.ipynb    # Full notebook (EDA -> model -> evaluation)
├── figures/
│   ├── fig_distributions.png
│   ├── fig_correlation.png
│   ├── fig_confusion_matrix.png
│   └── fig_feature_importance.png
└── README.md
```

## Future work

- Replace synthetic data with real ITRA / Google Earth Engine / Togo Meteorological
  data collected in Phase 1
- Extend to a multi-output system (planting timing via LSTM, disease detection via
  MobileNetV2 CNN / TinyML)
- Compare Random Forest vs XGBoost on real data
- Optimize for mobile deployment (TensorFlow Lite, < 50 MB, < 500 ms inference)

## License

Academic project — free use for research and educational purposes.
