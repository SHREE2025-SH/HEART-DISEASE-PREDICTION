import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Page config
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Prediction App")
st.markdown("Enter patient details to get predictions from **Logistic Regression** and **Decision Tree** models (trained on UCI Heart Disease data).")

# Load models and scaler (cached for speed)
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('outputs/logistic_regression_model.pkl')
        dt_model = joblib.load('outputs/decision_tree_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        st.success("Models loaded successfully!")
        return lr_model, dt_model, scaler
    except FileNotFoundError:
        st.error("PKL files not found in 'outputs/'. Run heart_model3.py first!")
        return None, None, None

lr_model, dt_model, scaler = load_models()

# Features (exact match to your script)
features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Sidebar with example and info
with st.sidebar:
    st.header("ℹ️ Quick Info")
    st.markdown("""
    - **Models:** Logistic Regression (scaled) vs Decision Tree (unscaled).
    - **Features:** Based on UCI dataset.
    - **Example Patient:** Use the values below for a sample prediction.
    """)
    st.subheader("📋 Sample Input")
    example = {
        'age': 55, 'sex': 1, 'cp': 2, 'trestbps': 140, 'chol': 250,
        'fbs': 0, 'restecg': 1, 'thalach': 150, 'exang': 0,
        'oldpeak': 1.5, 'slope': 1, 'ca': 0, 'thal': 2
    }
    use_example = st.checkbox("Load Sample Patient Data")
    if use_example:
        for feat in features:
            st.session_state[feat] = example[feat]

# Main input form (two columns for better layout)
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Demographics & Basics")
    age = st.slider("Age (29-77)", 29, 77, 50 if 'age' not in st.session_state else st.session_state['age'])
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1], index=1 if 'sex' not in st.session_state else st.session_state['sex'])
    cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4], index=1 if 'cp' not in st.session_state else st.session_state['cp'] - 1)
    trestbps = st.slider("Resting Blood Pressure (mm Hg, 94-200)", 94, 200, 130 if 'trestbps' not in st.session_state else st.session_state['trestbps'])
    chol = st.slider("Serum Cholesterol (mg/dl, 126-564)", 126, 564, 246 if 'chol' not in st.session_state else st.session_state['chol'])
    fbs = st.selectbox("Fasting Blood Sugar >120 mg/dl (1=Yes)", [0, 1], index=0 if 'fbs' not in st.session_state else st.session_state['fbs'])

with col2:
    st.subheader("🏥 Test Results")
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2], index=0 if 'restecg' not in st.session_state else st.session_state['restecg'])
    thalach = st.slider("Max Heart Rate (71-202)", 71, 202, 150 if 'thalach' not in st.session_state else st.session_state['thalach'])
    exang = st.selectbox("Exercise Induced Angina (1=Yes)", [0, 1], index=0 if 'exang' not in st.session_state else st.session_state['exang'])
    oldpeak = st.slider("ST Depression (0.0-6.2)", 0.0, 6.2, 1.0 if 'oldpeak' not in st.session_state else st.session_state['oldpeak'])
    slope = st.selectbox("ST Segment Slope (1-3)", [1, 2, 3], index=0 if 'slope' not in st.session_state else st.session_state['slope'] - 1)
    ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3], index=0 if 'ca' not in st.session_state else st.session_state['ca'])
    thal = st.selectbox("Thalassemia (1=Normal, 2=Fixed, 3=Reversible)", [1, 2, 3], index=0 if 'thal' not in st.session_state else st.session_state['thal'] - 1)

# Store in session state for persistence
for feat, val in zip(features, [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]):
    st.session_state[feat] = val

# Prepare input DataFrame
new_patient = pd.DataFrame({
    'age': [st.session_state['age']], 'sex': [st.session_state['sex']], 'cp': [st.session_state['cp']],
    'trestbps': [st.session_state['trestbps']], 'chol': [st.session_state['chol']], 'fbs': [st.session_state['fbs']],
    'restecg': [st.session_state['restecg']], 'thalach': [st.session_state['thalach']], 'exang': [st.session_state['exang']],
    'oldpeak': [st.session_state['oldpeak']], 'slope': [st.session_state['slope']], 'ca': [st.session_state['ca']],
    'thal': [st.session_state['thal']]
})

# Predict button
if st.button("🔮 Predict Risk", type="primary") and lr_model is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Logistic Regression")
        # Scale for LR
        new_patient_scaled = scaler.transform(new_patient)
        lr_pred = lr_model.predict(new_patient_scaled)[0]
        lr_prob = lr_model.predict_proba(new_patient_scaled)[0][1] * 100  # Disease prob
        
        st.write(f"**Prediction:** {'🩸 Heart Disease' if lr_pred == 1 else '✅ No Heart Disease'}")
        st.metric("Disease Probability", f"{lr_prob:.1f}%", delta=None)
        if lr_pred == 1:
            st.error("⚠️ High risk detected—recommend medical consultation!")
        else:
            st.success("✅ Low risk.")
    
    with col2:
        st.subheader("🌳 Decision Tree")
        # No scaling for DT
        dt_pred = dt_model.predict(new_patient)[0]
        dt_prob = dt_model.predict_proba(new_patient)[0][1] * 100  # Disease prob
        
        st.write(f"**Prediction:** {'🩸 Heart Disease' if dt_pred == 1 else '✅ No Heart Disease'}")
        st.metric("Disease Probability", f"{dt_prob:.1f}%", delta=None)
        if dt_pred == 1:
            st.error("⚠️ High risk detected—recommend medical consultation!")
        else:
            st.success("✅ Low risk.")
    
    # Agreement check
    if lr_pred == dt_pred:
        st.info(f"🤝 **Both models agree:** {'Heart Disease' if lr_pred == 1 else 'No Heart Disease'}")
    else:
        st.warning("❓ **Models disagree**—consider additional tests.")
