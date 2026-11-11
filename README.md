# 🌌 Astro-ML: Stellar Classification & Redshift Estimation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shikhar0512L/Astro-ML-Stellar-Classification-Redshift-Estimation/blob/main/AstroML.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning pipeline that classifies celestial objects (Stars, Galaxies, Quasars) from SDSS photometric data and attempts to predict their exact **Redshift**. The notebook documents not only the successes but also the **scientific limits** of photometric-only regression — including data leakage, the extrapolation ceiling, and the failure modes of naive feature engineering.

---

## 📖 What is this project?

Telescopes like the Sloan Digital Sky Survey (SDSS) observe hundreds of millions of celestial objects and record how much light they emit across different color bands. Manually labelling each object is impossible, so this project trains machine learning models to do it automatically.

The pipeline tackles two tasks:

1. **Classification** — Identify the object type: `STAR`, `GALAXY`, or `QSO` (Quasar).
2. **Regression** — Predict the object's **Redshift**, a proxy for cosmological distance and age.

Everything is contained in a single, self-documenting Jupyter notebook that explains each concept from first principles — including where the models break and why.

---

## 🧠 Core Concepts (Quick Glossary)

| Term | Meaning |
|---|---|
| **Redshift** | As the universe expands, light from distant objects stretches toward red. Higher redshift → farther and older object. |
| **Photometric data** | Brightness measured through 5 broad filters (`u, g, r, i, z`). Fast to capture, low resolution. |
| **Spectroscopic data** | Full wavelength-by-wavelength light split. Highly accurate, very slow. |
| **Data leakage** | Accidentally feeding the model the target column during training. Fixed explicitly in this project. |
| **Extrapolation ceiling** | Tree-based models cannot predict values outside the range seen during training. |

---

## 🏆 Results at a Glance

| Model | Task | Key Metric |
|---|---|---|
| Random Forest Classifier | Classification | **97.73% accuracy** |
| MLPRegressor (Neural Net) | Redshift regression | $R^2 = 0.4187$ · $\text{MSE} = 0.3073$ |
| XGBoost Regressor | Redshift regression | $R^2 = 0.4098$ · $\text{MSE} = 0.3121$ |
| XGBoost + Feature Engineering + Sample Weights | Redshift regression | $R^2 = 0.1295$ · $\text{MSE} = 0.4603$ |

> **TL;DR** — Classification is essentially solved. Regression hits a hard scientific ceiling (~redshift 2.5) because photometric features do not carry enough information to resolve high-redshift objects. Forcing the model to care about Quasars causes catastrophic over-correction.

---

## 📂 Repository Structure

```
.
├── AstroML.ipynb                # Full pipeline: EDA → classification → regression → conclusion
├── star_classification.csv      # SDSS dataset (~100,000 objects, 18 columns)
├── README.md
├── requirements.txt
└── LICENSE
```

---

## 🚀 How to Run

**Option 1 — Google Colab (recommended)**

Click the "Open in Colab" badge above, then `Runtime → Run all`. The dataset is fetched from a remote URL, so no local setup is needed.

**Option 2 — Local**

```bash
git clone https://github.com/shikhar0512L/Astro-ML-Stellar-Classification-Redshift-Estimation.git
cd Astro-ML-Stellar-Classification-Redshift-Estimation
pip install -r requirements.txt
jupyter notebook AstroML.ipynb
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Pandas / NumPy** — data loading, cleaning, feature engineering
- **Scikit-learn** — `RandomForestClassifier`, `MLPRegressor`, `StandardScaler`, `train_test_split`, metrics
- **XGBoost** — gradient-boosted regression baseline
- **Matplotlib / Seaborn** — confusion matrix, residual plots
- **Google Colab** — execution environment

---

## ✅ Phase 1: Classification (Success)

**Goal:** Predict `STAR` / `GALAXY` / `QSO` from the 5 photometric bands.

**Approach:**
- Dropped non-predictive ID columns (`obj_ID`, `run_ID`, `MJD`, `plate`, etc.)
- Standardized all features with `StandardScaler`
- Encoded the target with `LabelEncoder`
- Trained a **Random Forest Classifier** (100 trees, voting ensemble)
- 80/20 train-test split

**Result:** **97.73% accuracy**

The confusion matrix shows the model occasionally confuses **Quasars with Galaxies** — which is scientifically expected, since quasars are active galactic nuclei embedded within galaxies and their photometric signatures overlap heavily.

---

## 📉 Phase 2: Regression — The Redshift Bottleneck

**Goal:** Predict the exact numerical redshift value for each object.

### 🔧 The Data Leakage Fix

The first regression attempt reported a suspiciously perfect $R^2 = 1.0$. Root cause: the `redshift` column was still present in the input features $X$ — the model was literally reading the answer key.

**Fix:** Explicitly dropped `redshift` and `class` from $X$. The realistic $R^2$ then dropped to ~0.42, which is the honest baseline for this dataset.

### 🧪 Models Tested

| Model | Result | Why It Failed |
|---|---|---|
| **MLPRegressor** `(100, 50)` | $R^2 = 0.42$ | Predicted physically impossible **negative redshifts**; high variance near the ceiling. |
| **XGBoost Regressor** `(200 trees, depth 6)` | $R^2 = 0.41$ | Same extrapolation ceiling as any tree-based model. |
| **XGBoost + Color Indices + Sample Weights** | $R^2 = 0.13$ | **Over-correction** — the $10\times$ penalty on high-redshift samples caused the model to hallucinate high redshifts for normal galaxies. |

### 📊 The Extrapolation Ceiling

All tree-based models flatline at **redshift ≈ 2.5** because:

- 90% of the dataset is low-redshift.
- A decision tree physically cannot output a value higher than the maximum seen in training.
- The model optimizes for MSE, so "guess 1.5 for everything" is near-optimal for 90% of the data — at the cost of completely missing Quasars.

Feature engineering (color indices `u-g`, `g-r`, `r-i`, `i-z`) and sample weighting were applied as a corrective, but they **made the model worse**, not better — a clean demonstration that when the data is information-poor, more aggressive training makes things worse, not better.

---

## 🏁 Scientific Conclusion

The failure is not a modelling failure — it is a **data limitation**.

Predicting precise redshift from 5 broad photometric bands is analogous to guessing a person's exact weight from their shoe size. The signal is simply not present in the features.

To break the redshift ≈ 2.5 ceiling, the pipeline must move from **photometric CSV data** to **spectroscopic FITS data**, and from **tabular models** to **1D Convolutional Neural Networks**, which are designed to find patterns in continuous signals like light spectra.

This is the point where the project transitions from a tabular ML problem into a domain-specific astrophysics ML problem.

---

## 🗺️ Roadmap — Phase 2

- [ ] Ingest spectroscopic FITS files from SDSS DR18
- [ ] Build a preprocessing pipeline for raw 1D spectra
- [ ] Train a 1D-CNN for redshift regression
- [ ] Benchmark against template-fitting baselines (e.g. Redrock)
- [ ] Add per-class error analysis (Star vs Galaxy vs QSO)
- [ ] Deploy an interactive inference demo (Gradio / Streamlit)


---

## 📦 Dataset

- **Source:** [SDSS Stellar Classification Dataset](https://raw.githubusercontent.com/dannycab/MSU_REU_ML_course/main/activities/data/star_classification.csv) — 100,000 rows, 18 columns.
- The notebook loads the CSV directly from a remote URL, so the file is optional for reproduction. It is included in this repository for archival completeness.

---

## 📜 License

Released under the [MIT License](LICENSE).

---

## 👤 Author

**Shikhar Singh**  
B.Tech CSE (AI & ML) — KIIT University  
[GitHub](https://github.com/shikhar0512L)

If this project helped you understand ML extrapolation limits or photometric redshift estimation, consider giving it a ⭐.
