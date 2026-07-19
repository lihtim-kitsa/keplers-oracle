import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import src.config as config

# --- Configuration & Setup ---
st.set_page_config(
    page_title="Kepler's Oracle",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load resources
@st.cache_resource
def load_models():
    # Load Ensemble Model
    model = joblib.load(config.MODEL_DIR / 'best_xgb_model.joblib')
    scaler = joblib.load(config.MODEL_DIR / 'scaler.joblib')
    imputer = joblib.load(config.MODEL_DIR / 'imputer.joblib')
    label_encoder = joblib.load(config.MODEL_DIR / 'label_encoder.joblib')
    feature_names = joblib.load(config.MODEL_DIR / 'feature_names.joblib')
    return model, scaler, imputer, label_encoder, feature_names

@st.cache_data
def load_dataset():
    df = pd.read_csv(config.DATA_PATH)
    df = df.dropna(subset=[config.TARGET_COL])
    # Recalculate engineered features for display purposes
    if 'koi_teq' in df.columns:
        df['goldilocks_flag'] = ((df['koi_teq'] > 273) & (df['koi_teq'] < 373)).astype(int)
    else:
        df['goldilocks_flag'] = 0
    return df

try:
    model, scaler, imputer, le, feature_names = load_models()
    df = load_dataset()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models or data. Make sure to run the training pipeline first. Details: {e}")
    models_loaded = False

# --- App Layout ---

st.sidebar.title("🔭 Kepler's Oracle")
st.sidebar.markdown("""
Welcome to **Kepler's Oracle**! 

This advanced interactive dashboard uses an Ensemble Machine Learning model (XGBoost, Random Forest, LightGBM) to classify potential exoplanets from NASA's Kepler data.
""")
page = st.sidebar.radio("Navigate", [
    "Project Overview", 
    "The Goldilocks Zone", 
    "Data Explorer", 
    "Prediction Simulator", 
    "Model Showdown",
    "Explainable AI (SHAP)"
])

if page == "Project Overview":
    st.title("Are We Alone in the Universe?")
    st.markdown("""
    The discovery of planets beyond our solar system has always fascinated humanity. Every confirmed exoplanet brings us one step closer to answering our biggest questions. 
    
    When NASA made the Kepler Objects of Interest (KOI) dataset publicly available, it opened the door for citizen scientists and data enthusiasts to explore the cosmos. That curiosity inspired the creation of **Kepler's Oracle**.
    
    ### What does this project do?
    Kepler's Oracle is a state-of-the-art machine learning pipeline and interactive dashboard that classifies potential exoplanets into three categories:
    - 🟢 **CONFIRMED**: Validated as a real planet.
    - 🟡 **CANDIDATE**: A transit-like signal not yet confirmed.
    - 🔴 **FALSE POSITIVE**: A signal explained by noise, eclipsing binaries, or other artifacts.
    
    ### The Expansion
    This project features several advanced techniques:
    - **SMOTE (Synthetic Minority Over-sampling Technique):** To handle the imbalanced nature of astronomical data, ensuring rare classes are not ignored.
    - **Ensemble Modeling (Voting Classifier):** Combining XGBoost, Random Forest, and LightGBM for maximum predictive power.
    - **Astrophysics Feature Engineering:** Creating new variables like the *Habitability Index* to inject real physics into the ML models.
    - **Explainable AI (SHAP):** Ensuring the models aren't "black boxes" by visually proving how they make decisions.
    
    Explore the other tabs to dive deep into the cosmos!
    """)
    st.image("https://exoplanets.nasa.gov/internal_resources/1922/", caption="Artist's concept of a planetary system. (Credit: NASA)")

elif page == "The Goldilocks Zone":
    if not models_loaded:
        st.stop()
    
    st.title("🌍 The Goldilocks Zone")
    st.markdown("""
    In astronomy, the habitable zone (or "Goldilocks Zone") is the range of orbits around a star within which a planetary surface can support liquid water given sufficient atmospheric pressure.
    
    We engineered a specific feature (`goldilocks_flag`) using the Equilibrium Temperature (`koi_teq`) to proxy this zone (approx. 273K - 373K).
    """)
    
    # Filter to only confirmed and candidates that have teq and radius
    df_plot = df[(df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])) & (df['koi_prad'] < 20)].copy()
    
    fig = px.scatter(df_plot, x='koi_teq', y='koi_prad', 
                     color='goldilocks_flag',
                     color_discrete_map={1: "dodgerblue", 0: "gray"},
                     hover_data=["kepoi_name", "koi_disposition"],
                     title="Planetary Radius vs. Equilibrium Temp (Goldilocks Planets Highlighted)",
                     labels={"koi_teq": "Equilibrium Temperature (K)", "koi_prad": "Planetary Radius (Earth radii)", "goldilocks_flag": "In Habitable Zone?"})
    
    # Add vertical lines for the zone
    fig.add_vline(x=273, line_dash="dash", line_color="green", annotation_text="0°C (Freezing)")
    fig.add_vline(x=373, line_dash="dash", line_color="red", annotation_text="100°C (Boiling)")
    
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("*Note: This is a simplified proxy. Actual habitability depends on atmospheric composition, albedo, and stellar type.*")

elif page == "Data Explorer":
    if not models_loaded:
        st.stop()
        
    st.title("📊 Data Explorer")
    st.markdown("Explore the raw Kepler dataset to understand the relationships between different planetary and stellar features.")
    
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-Axis", feature_names, index=feature_names.index("koi_period") if "koi_period" in feature_names else 0)
    with col2:
        y_axis = st.selectbox("Y-Axis", feature_names, index=feature_names.index("koi_prad") if "koi_prad" in feature_names else 1)
        
    # Filtering for better visualization (removing extreme outliers)
    if x_axis in df.columns and y_axis in df.columns:
        q_low_x = df[x_axis].quantile(0.01)
        q_hi_x  = df[x_axis].quantile(0.95)
        q_low_y = df[y_axis].quantile(0.01)
        q_hi_y  = df[y_axis].quantile(0.90) 
        
        df_filtered = df[(df[x_axis] < q_hi_x) & (df[x_axis] > q_low_x) & 
                         (df[y_axis] < q_hi_y) & (df[y_axis] > q_low_y)]

        fig = px.scatter(df_filtered, x=x_axis, y=y_axis, color="koi_disposition", 
                         color_discrete_map={"CONFIRMED": "green", "CANDIDATE": "yellow", "FALSE POSITIVE": "red"},
                         hover_data=["kepoi_name"],
                         title=f"{y_axis} vs {x_axis} (Outliers Filtered)",
                         opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selected features are not available in the raw dataset (they might be engineered).")

elif page == "Prediction Simulator":
    if not models_loaded:
        st.stop()
        
    st.title("🎛️ Exoplanet Prediction Simulator")
    st.markdown("Tweak the astrophysical properties below to see how the **Ensemble Model** predicts the disposition of the object in real-time.")
    
    st.sidebar.markdown("### Simulator Controls")
    
    # Create input sliders for features based on median and IQR of real data
    input_data = {}
    for feature in config.FEATURES: # Only raw features
        median_val = df[feature].median()
        min_val = float(df[feature].quantile(0.05))
        max_val = float(df[feature].quantile(0.95))
        
        if min_val == max_val:
            min_val = min_val - 1.0
            max_val = max_val + 1.0
            
        input_data[feature] = st.sidebar.slider(
            feature, 
            min_value=min_val, 
            max_value=max_val, 
            value=float(median_val),
            format="%.2f"
        )
        
    # Recalculate engineered features for simulator input
    input_df = pd.DataFrame([input_data])
    
    # 1. Goldilocks Flag
    if 'koi_teq' in input_df.columns:
        input_df['goldilocks_flag'] = ((input_df['koi_teq'] > 273) & (input_df['koi_teq'] < 373)).astype(int)
    else:
        input_df['goldilocks_flag'] = 0
        
    # 2. Planet Size Class
    if 'koi_prad' in input_df.columns:
        conditions = [
            (input_df['koi_prad'] < 1.25),
            (input_df['koi_prad'] >= 1.25) & (input_df['koi_prad'] < 2.0),
            (input_df['koi_prad'] >= 2.0) & (input_df['koi_prad'] < 4.0),
            (input_df['koi_prad'] >= 4.0)
        ]
        choices = [1, 2, 3, 4]
        input_df['planet_size_class'] = np.select(conditions, choices, default=0)
    else:
        input_df['planet_size_class'] = 0

    # Ensure column order matches training data
    input_df = input_df[feature_names]
    
    input_imputed = imputer.transform(input_df)
    input_scaled = scaler.transform(input_imputed)
    
    # Predict using the Ensemble model
    prediction_proba = model.predict_proba(input_scaled)[0]
    prediction_idx = np.argmax(prediction_proba)
    prediction_label = le.inverse_transform([prediction_idx])[0]
    
    # Display Result
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Prediction")
        color_map = {"CONFIRMED": "green", "CANDIDATE": "orange", "FALSE POSITIVE": "red"}
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid {color_map.get(prediction_label, 'white')};">
            <h1 style='color: {color_map.get(prediction_label, 'white')}; margin: 0;'>{prediction_label}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Generated Planet Profile
        st.markdown("### 🪐 Generated Profile")
        radius = input_df['koi_prad'].values[0]
        teq = input_df['koi_teq'].values[0]
        
        size_str = "Earth-like" if radius < 1.25 else "Super-Earth" if radius < 2.0 else "Mini-Neptune" if radius < 4.0 else "Giant/Jupiter-like"
        temp_str = "Habitable" if (273 < teq < 373) else "Scorching" if teq >= 373 else "Freezing"
        
        st.info(f"**Classification:** {temp_str} {size_str}")

    with col2:
        st.subheader("Ensemble Probabilities")
        prob_df = pd.DataFrame({
            'Class': le.classes_,
            'Probability': prediction_proba * 100
        })
        
        fig = px.bar(prob_df, x='Class', y='Probability', 
                     color='Class', 
                     color_discrete_map={"CONFIRMED": "green", "CANDIDATE": "orange", "FALSE POSITIVE": "red"},
                     text_auto='.1f',
                     title="Confidence by Class (%)")
        fig.update_layout(yaxis_range=[0, 100], margin=dict(t=30, l=0, r=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Showdown":
    st.title("🏆 Model Showdown")
    st.markdown("""
    Instead of relying on a single algorithm, Kepler's Oracle trained three state-of-the-art models and evaluated them. 
    We then combined them into a **Voting Classifier** (Ensemble) to maximize accuracy and robustness.
    
    Below is the Multi-Class Receiver Operating Characteristic (ROC) curve for the final Ensemble Model.
    """)
    
    try:
        st.image(str(config.RESULTS_DIR / 'roc_curve.png'), caption="Ensemble Model ROC Curve")
    except FileNotFoundError:
        st.warning("ROC curve plot not found. Run the training pipeline.")
        
    st.markdown("""
    **Understanding the ROC Curve:**
    - The top-left corner is the "ideal" spot (100% True Positive Rate, 0% False Positive Rate).
    - The closer the curves (and their Area Under Curve - AUC) are to 1.0, the better the model is at distinguishing that specific class.
    """)

elif page == "Explainable AI (SHAP)":
    st.title("🧠 Model Insights (Explainable AI)")
    st.markdown("""
    Machine Learning models are often criticized for being "black boxes". To ensure Kepler's Oracle is scientifically rigorous, we use **SHAP** (SHapley Additive exPlanations) to explain how the model makes decisions.
    """)
    
    st.subheader("Confusion Matrix")
    st.markdown("This shows how well the ensemble model performed on the hold-out test data. The diagonal represents correct predictions.")
    try:
        st.image(str(config.RESULTS_DIR / 'confusion_matrix.png'), width=600)
    except FileNotFoundError:
        st.warning("Confusion matrix plot not found. Run the training pipeline.")
        
    st.subheader("Global Feature Importance (SHAP)")
    st.markdown("The SHAP summary plot shows which features most strongly impact the predictions. We use the XGBoost base estimator for this explanation.")
    try:
        st.image(str(config.RESULTS_DIR / 'shap_summary.png'))
    except FileNotFoundError:
        st.warning("SHAP summary plot not found. Run the training pipeline.")
        
    st.markdown("""
    **Key Takeaways:**
    * **`koi_prad` (Planetary Radius):** Often the most important feature. If the radius is extremely large, the model strongly suspects a FALSE POSITIVE (likely a companion star, not a planet).
    * **`koi_depth` (Transit Depth):** The percentage of light blocked. Crucial for determining if a transit is real.
    """)
