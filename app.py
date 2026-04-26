import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "diabetes_model_final.pkl"
FINAL_THRESHOLD = 0.9

st.set_page_config(
    page_title="Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide"
)

# -------------------- CSS --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f6f9ff 0%, #f2fbf7 45%, #fff7f9 100%);
    color: #101828;
}
.block-container {padding-top: 1.2rem;}

h1, h2, h3, h4 { color: #101828; }
p, li, span, label { color: #344054; }

.title {
    font-size: 34px;
    font-weight: 900;
    margin-bottom: 6px;
}
.subtitle { opacity: 0.9; margin-top: -4px; }

.badge {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 12px;
    background: rgba(45, 124, 255, 0.10);
    border: 1px solid rgba(45, 124, 255, 0.25);
    color: #1d4ed8;
}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 900;
    padding: 0.75rem 1rem;
    background: linear-gradient(135deg, #2d7cff 0%, #00d4ff 100%);
    color: #ffffff;
}

.big-number {
    font-size: 42px;
    font-weight: 950;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------- Risk Function --------------------
def risk_bucket(p: float):
    if p < 0.4:
        return "Low risk"
    elif p < 0.6:
        return "Medium risk"
    elif p < 0.9:
        return "High risk"
    else:
        return "Very high risk"

# -------------------- Header --------------------
st.markdown('<div class="title">🩺 Diabetes Risk Assessment (ML)</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">This system does not diagnose, only estimates risk. Threshold: <span class="badge">{FINAL_THRESHOLD}</span></div>',
    unsafe_allow_html=True
)

left, right = st.columns([1.05, 0.95])

# -------------------- INPUT --------------------
with left:
    st.subheader("📌 Enter Information")

    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=1, max_value=120, value=35)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
        hba1c = st.number_input("HbA1c level", min_value=3.0, max_value=15.0, value=5.5)

    with c2:
        smoking_history = st.selectbox(
            "Smoking history",
            ["never", "No Info", "current", "former", "ever", "not current"]
        )

        hypertension_label = st.selectbox("Hypertension", ["❌ No", "✅ Yes"])
        heart_disease_label = st.selectbox("Heart disease", ["❌ No", "✅ Yes"])

        # Updated logic
        hypertension = 1 if hypertension_label.endswith("Yes") else 0
        heart_disease = 1 if heart_disease_label.endswith("Yes") else 0

        glucose = st.number_input("Blood glucose level", min_value=50.0, max_value=400.0, value=110.0)

    run = st.button("🔍 Predict")

# -------------------- OUTPUT --------------------
with right:
    st.subheader("📊 Result")

    if run:
        input_df = pd.DataFrame([{
            "gender": gender,
            "age": float(age),
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "smoking_history": smoking_history,
            "bmi": float(bmi),
            "HbA1c_level": float(hba1c),
            "blood_glucose_level": float(glucose)
        }])

        proba = float(model.predict_proba(input_df)[0, 1])
        final_class = int(proba >= FINAL_THRESHOLD)

        st.markdown(f"<p class='big-number'>{proba:.4f}</p>", unsafe_allow_html=True)
        st.write(f"Risk: {risk_bucket(proba)}")

        st.progress(proba)

        st.subheader("📌 Final conclusion")
        if final_class == 1:
            st.error("Very high probability detected (P ≥ 0.9)")
        else:
            st.success("Below critical threshold (monitoring recommended)")

        st.write("### Input data:")
        st.dataframe(input_df, use_container_width=True)

    else:
        st.info("Enter data and click Predict.")
