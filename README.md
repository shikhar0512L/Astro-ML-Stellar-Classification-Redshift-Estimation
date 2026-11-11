# 🌌 Astro-ML: Stellar Classification & Redshift Estimation

## 📖 What is this project? (From Scratch)
Imagine looking up at the night sky. There are billions of stars, galaxies, and mysterious objects called Quasars. Telescopes like NASA's SDSS (Sloan Digital Sky Survey) take pictures of these objects and record how much light they emit in different colors. 

Because there are millions of these objects, humans cannot look at every single one and say, *"This is a star"* or *"This is a galaxy."* 

**This project uses Artificial Intelligence (Machine Learning) to automate this process.** 
We feed the AI the "colors of light" (data) from 100,000 space objects. The AI learns the patterns and does two things:
1. **Classification:** Tells us exactly what the object is (Star, Galaxy, or Quasar).
2. **Regression:** Predicts the exact distance of the object from Earth (measured by something called "Redshift").

This repository is a complete, step-by-step Jupyter Notebook that explains everything from scratch, including the successes, the failures, and the scientific limits of AI.

---

## 🧠 Core Concepts Explained (For Beginners)

Before looking at the code, here is what you need to know:
*   **What is Redshift?** As the universe expands, objects move away from us. As they move away, their light stretches and turns red. The higher the Redshift number, the farther away (and older) the object is.
*   **Photometric Data (What we used):** This is like looking at an object through 5 broad colored glass filters (`u`, `g`, `r`, `i`, `z` bands). It's fast but not very detailed.
*   **Spectroscopic Data (What we didn't use):** This is splitting the light into a detailed rainbow (thousands of specific wavelengths). It's highly accurate but very slow to capture.
*   **Data Leakage:** A beginner mistake where you accidentally give the AI the "answer key" during training. We specifically fixed this in the code.

---

## 🛠️ Tech Stack & Libraries
Here is exactly what we used and why:
*   **Python:** The programming language.
*   **Pandas:** Used to load the massive CSV file and clean the data (like finding missing values).
*   **Matplotlib & Seaborn:** Used to draw charts and the "Confusion Matrix" to see where the AI made mistakes.
*   **Scikit-Learn:** The main Machine Learning library. We used it for `RandomForestClassifier`, `MLPRegressor` (Neural Network), `StandardScaler` (scaling numbers so the AI doesn't get confused by big vs small numbers), and `train_test_split` (splitting data into 80% for studying and 20% for an exam).
*   **XGBoost:** The "Hedge Fund algorithm," known for being extremely fast and accurate on tabular data.
*   **Google Colab:** The cloud platform used to run the code for free.

---

## 🚀 Phase 1: The Classification (Success)

### The Goal
To teach the AI to look at the 5 light bands (`u, g, r, i, z`) and predict if the object is a **STAR**, **GALAXY**, or **QSO (Quasar)**.

### How we did it
1.  **Data Cleaning:** We dropped useless columns like `obj_ID` (just a serial number) and `MJD` (dates).
2.  **Scaling:** We used `StandardScaler` to make sure all the light values were on a similar scale.
3.  **The Algorithm:** We used a **Random Forest Classifier**. Imagine this as 100 different people (Decision Trees) looking at the data, making a guess, and then voting on the final answer.

### The Result
*   **97.73% Accuracy!** The AI was incredibly accurate.
*   **The Confusion Matrix:** When we visualized the results, we saw the AI sometimes confused a Quasar with a Galaxy. *Why?* Because a Quasar is actually a supermassive black hole eating a galaxy. Their light looks very similar to the AI! This is scientifically accurate.

---

## 📉 Phase 2: The Regression (The Brutal Reality)

### The Goal
To predict the exact **Redshift number** (e.g., 0.543, 2.1, 5.6). This is much harder than classification because we are predicting a continuous number, not just a category.

### The Crucial Mistake We Fixed (Data Leakage)
In our first attempt, our model got a perfect R2 score of 1.0 (100% accuracy). We quickly realized we made a mistake: we accidentally left the `redshift` column inside the input features. The AI was literally just reading the answer off the answer key!
**Fix:** We dropped the `redshift` column from the inputs (`X`) and made it the target (`y`). The score dropped to a realistic level.

### The Models We Tested & Why They Failed
1.  **Random Forest Regressor:** It failed to predict any Redshift higher than ~2.5. *Why?* Because a Decision Tree physically cannot guess a number higher than the highest number it saw in training. 90% of our data was low-redshift. So, it just guessed around 1.5 for everything.
2.  **Neural Network (MLPRegressor):** This is an "Artificial Brain." We gave it 2 hidden layers (100, 50). It predicted negative Redshift values (which is impossible in physics!) and performed poorly on high redshifts.
3.  **XGBoost:** We brought out the "Hedge Fund King." It built 200 trees sequentially, learning from mistakes. It was fast, but it still hit the same "extrapolation ceiling." It couldn't guess what it hadn't seen.

### The Final Experiment (Feature Engineering & Over-correction)
To fix this, we did two advanced things:
1.  **Feature Engineering:** We created new "Color Indices" by subtracting the light bands (e.g., `u-g`, `g-r`). Astronomers do this to find hidden signatures in the light.
2.  **Sample Weighting:** We told the AI, *"If you see a high Redshift (>2.0), it is rare, so pay 10x more attention to it."*

**The Result:** A disaster. The model **over-corrected**. It became so paranoid about missing high redshifts that it started hallucinating high redshifts for normal galaxies. The R2 score crashed to 0.13.

---

## 🏁 Final Conclusion & The Scientific Limit

Why did all these powerful algorithms fail to predict high redshifts accurately? 

**Answer:** Because we used **Photometric Data** (5 broad colors). 
Trying to predict the exact distance of a Quasar using just 5 broad colors is like trying to guess a person's exact weight by only looking at their shoe size. The clues are not rich enough. 

To break the ~2.5 Redshift ceiling, we cannot use CSV data. We must move to **Spectroscopic FITS files** (raw, detailed light waves). And to analyze raw waves, we need a completely different type of AI called **1D-CNNs (1D Convolutional Neural Networks)**.

---
