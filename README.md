# 🌌 Astro-ML: Stellar Classification & Redshift Estimation

## 📖 Overview
Astronomical observatories generate petabytes of data daily. Manual classification of celestial objects is impossible. This project builds a complete Machine Learning pipeline using the **Sloan Digital Sky Survey (SDSS)** dataset to automate two critical tasks:
1. **Classification:** Identifying whether an object is a Star, Galaxy, or Quasar (QSO).
2. **Regression:** Predicting the exact numerical value of an object's Redshift (which indicates its distance, age, and how fast it is moving away from us).

This repository documents not just the successful models, but also the **scientific limitations** discovered during the experimentation process when dealing with noisy, imbalanced space data.

## 📊 Dataset Information
The dataset consists of **100,000 rows** and **18 columns** obtained from the SDSS. 
*   **Features:** Photometric broad-band colors (`u`, `g`, `r`, `i`, `z`), coordinates (`alpha`, `delta`), and other observational metadata.
*   **Target Variables:** 
    *   `class` (Categorical: GALAXY, STAR, QSO)
    *   `redshift` (Continuous numerical value)

## ⚙️ Methodology & Workflow

### Phase 1: Object Classification (Random Forest)
We trained a **Random Forest Classifier** to categorize the celestial bodies based on their photometric light signatures.
*   **Preprocessing:** Dropped irrelevant ID columns, encoded target labels, and applied `StandardScaler` for normalization.
*   **Result:** Achieved a highly accurate **97.73% accuracy**. 
*   **Observation:** The confusion matrix revealed that the model occasionally confused Quasars with Galaxies. This is scientifically expected, as Quasars are essentially active galactic nuclei (a type of galaxy) with highly overlapping spectral features.

### Phase 2: Redshift Regression (The Bottleneck)
Predicting the exact distance (Redshift) proved to be significantly more challenging. We tested multiple algorithms:
1.  **Random Forest Regressor:** Failed to extrapolate beyond a redshift of ~2.5. (Decision trees cannot predict numbers higher than what they saw in training).
2.  **Neural Network (MLPRegressor):** Showed non-physical anomalies (predicting negative redshift) due to high variance.
3.  **XGBoost:** Did not solve the extrapolation ceiling.
4.  **Feature Engineering & Sample Weighting:** We created new features (Color Indices: `u-g`, `g-r`, `r-i`, `i-z`) and applied a 10x sample weight to rare, high-redshift Quasars. 
    *   *Result:* This caused a severe **over-correction**. The R-squared score dropped to 0.13, and the model began hallucinating high redshift values for normal galaxies.

## 🧠 Key Scientific Conclusion
The fundamental limitation of this dataset is that it relies on **Photometric data** (5 broad color filters) rather than full **Spectroscopic data** (raw wavelengths). 

Our experiments prove that tree-based models and simple neural networks hit an "extrapolation ceiling" at Redshift ~2.5 because 90% of the dataset consists of low-redshift objects. AI cannot guess the unknown without physical constraints.

## 🚀 Future Scope (Phase 2)
To accurately predict high-redshift Quasars, this project must evolve from tabular CSV data to raw **FITS files** containing full Spectroscopic data. Furthermore, to process raw continuous spectra, we will need to implement **1D Convolutional Neural Networks (1D-CNNs)**.

## 🛠️ Tech Stack
*   **Language:** Python 3
*   **Data Manipulation:** Pandas, NumPy
*   **Machine Learning:** Scikit-Learn, XGBoost
*   **Data Visualization:** Matplotlib, Seaborn
*   **Environment:** Google Colab

## 💻 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/Astro-ML.git
