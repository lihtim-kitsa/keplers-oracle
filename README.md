# Kepler's Oracle

> *The discovery of planets beyond our solar system has always fascinated humanity. Every confirmed exoplanet brings us one step closer to answering one of humanity's biggest questions: Are we alone in the universe? When I learned that NASA has made real exoplanet data publicly available, I wanted to explore how artificial intelligence could help analyze this enormous amount of information. That curiosity inspired me to build Kepler's Oracle.*

**Kepler's Oracle** is an advanced machine learning project and interactive dashboard that classifies potential exoplanets using real NASA datasets (Kepler Objects of Interest). 

This project goes beyond simple scripts—it provides a fully interactive web application where users can explore the cosmos, visualize data, and simulate exoplanet predictions in real-time.

---

## Tech Stack

- **Frontend / Dashboard:** Streamlit, Plotly
- **Machine Learning Core:** Scikit-learn, XGBoost, LightGBM
- **Data Processing:** Pandas, NumPy, Imbalanced-learn (SMOTE)
- **Explainable AI:** SHAP (SHapley Additive exPlanations)
- **Language:** Python 3.9+

## Features

- **Robust Data Pipeline:** Cleans and preprocesses raw sensor data, handles missing values, and selects the most relevant astrophysical features.
- **Ensemble Machine Learning:** Uses **XGBoost** (eXtreme Gradient Boosting) optimized via `RandomizedSearchCV` for high-accuracy multi-class predictions (`CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`).
- **Explainable AI (SHAP):** Breaks open the "black box" of the model. It provides clear, visual explanations of *why* the model makes its decisions, highlighting the importance of features like Planetary Radius (`koi_prad`) and Transit Depth (`koi_depth`).
- **Interactive Dashboard:** A beautifully designed Streamlit web application that lets you explore the data visually and tweak a live prediction simulator.

## Setup Instructions

1. **Install Dependencies:**
   Ensure you have Python 3.9+ installed. Run the following command in the project root:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ML Pipeline:**
   Before launching the app, you need to train the model and generate the evaluation artifacts.
   ```bash
   python src/data_prep.py
   python src/train.py
   python src/evaluate.py
   ```

3. **Launch the Dashboard:**
   Start the interactive Streamlit application:
   ```bash
   streamlit run app.py
   ```

## The Journey & Challenges

Building this project taught me much more than just coding. I learned how to work with real scientific datasets, prepare data for machine learning, and compare different classification algorithms. 

One of the biggest challenges was working with real-world data from the Kepler pipeline. The dataset contained missing values and complex features that required careful preprocessing. Choosing the right algorithm (XGBoost) and improving the model's performance required several rounds of hyperparameter tuning and testing. 

This project is not just about predicting exoplanets—it is about demonstrating how AI can support scientific discovery. I hope Kepler's Oracle inspires more students to explore the intersection of astronomy, data science, and artificial intelligence, and shows how technology can help us better understand our universe.

---
*Created for the Celesta India High School Exoplanet Data Challenge.*
