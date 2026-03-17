import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "diabetes_model_final.pkl"
FINAL_THRESHOLD = 0.9

st.set_page_config(
    page_title="Diabet Risk Baholash",
    page_icon="🩺",
    layout="wide"
)

# -------------------- CSS (Light / Professional + container-card fix) --------------------
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
    letter-spacing: 0.2px;
    margin-bottom: 6px;
    color: #101828;
}
.subtitle { opacity: 0.9; margin-top: -4px; color: #475467; }

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

/* Risk pills */
.pill-low  {background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.30); color:#166534;}
.pill-mid  {background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.25); color:#1e40af;}
.pill-high {background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.30); color:#92400e;}
.pill-vhigh{background: rgba(239,68,68,0.12); border: 1px solid rgba(239,68,68,0.30); color:#991b1b;}

.stButton>button {
    width: 100%;
    border-radius: 14px;
    font-weight: 900;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(16,24,40,0.12);
    background: linear-gradient(135deg, #2d7cff 0%, #00d4ff 100%);
    color: #ffffff;
}
.stButton>button:hover {filter: brightness(1.04);}

div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.95) !important;
    border: 1px solid rgba(16,24,40,0.12) !important;
}

.big-number {
    font-size: 42px;
    font-weight: 950;
    margin: 0;
    color: #101828;
}
.small-note { opacity: 0.85; font-size: 13px; color:#475467; }

hr {border: none; border-top: 1px solid rgba(16,24,40,0.10); margin: 12px 0;}

/* Tavsiya card */
.reco {
    border-radius: 14px;
    padding: 14px 14px;
    margin-top: 10px;
    border: 1px solid rgba(16,24,40,0.12);
    box-shadow: 0 8px 20px rgba(16,24,40,0.06);
}
.reco-title {
    font-weight: 900;
    margin: 0 0 6px 0;
    color: #101828;
    display: flex;
    gap: 8px;
    align-items: center;
}
.reco-text {
    margin: 0;
    color: #475467;
    line-height: 1.45;
}
.reco-low  { border-left: 6px solid rgba(34,197,94,0.9);  background: rgba(34,197,94,0.08); }
.reco-mid  { border-left: 6px solid rgba(59,130,246,0.8); background: rgba(59,130,246,0.08); }
.reco-high { border-left: 6px solid rgba(245,158,11,0.9); background: rgba(245,158,11,0.08); }
.reco-vhigh{ border-left: 6px solid rgba(239,68,68,0.9);  background: rgba(239,68,68,0.08); }

/* Streamlit container(border=True) ni "card" ko‘rinishiga keltiramiz */
div[data-testid="stVerticalBlockBorderWrapper"]{
    background: rgba(255,255,255,0.92) !important;
    border: 1px solid rgba(16,24,40,0.10) !important;
    border-radius: 18px !important;
    padding: 18px 18px 14px 18px !important;
    box-shadow: 0 12px 30px rgba(16,24,40,0.08) !important;
}

header {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- Model yuklash --------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------- Risk kategoriyalari (4 ta) + Tavsiya --------------------
def risk_bucket(p: float):
    if p < 0.4:
        return ("🟢 Past risk", "pill-low", "reco-low",
                "✅ Tavsiya",
                "Profilaktika va sog‘lom turmush tarzini davom ettiring: me’yoriy ovqatlanish, "
                "jismoniy faollik, vazn nazorati va davriy tekshiruvlar.")
    elif p < 0.6:
        return ("🔵 O‘rta risk", "pill-mid", "reco-mid",
                "📝 Tavsiya",
                "Risk o‘rta darajada. Ovqatlanish va jismoniy faollikni yaxshilang, vaznni nazorat qiling. "
                "1–3 oy ichida qayta tekshiruv (HbA1c/glyukoza) va shifokor maslahati tavsiya etiladi.")
    elif p < 0.9:
        return ("🟠 Yuqori risk", "pill-high", "reco-high",
                "🧑‍⚕️ Tavsiya",
                "Risk yuqori. Shifokor ko‘rigidan o‘tish tavsiya etiladi. "
                "Laborator tekshiruvlar (HbA1c, qon glyukozasi)ni yaqin muddatda topshiring. "
                "Turmush tarzini (ovqatlanish, faollik) keskinroq yaxshilash muhim.")
    else:
        return ("🔴 Juda yuqori risk", "pill-vhigh", "reco-vhigh",
                "🚨 Tavsiya",
                "Juda yuqori risk aniqlandi. Shifokorga tezroq murojaat qiling va laborator tekshiruvlarni "
                "kechiktirmang (HbA1c, qon glyukozasi va boshqalar). Zarur bo‘lsa davolash rejasi shifokor "
                "tomonidan belgilanadi.")

# -------------------- Header --------------------
st.markdown('<div class="title">🩺 Diabet riskini ML orqali baholash</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="subtitle">Eslatma: bu tizim <b>tashxis qo‘ymaydi</b>, faqat ehtimollik (risk) baholaydi. '
    f'Final threshold: <span class="badge">{FINAL_THRESHOLD}</span></div>',
    unsafe_allow_html=True
)
st.caption("⚙️ Final klass: model P ≥ 0.9 bo‘lsa — 1 (juda yuqori ehtimol), aks holda 0.")
st.write("")

# -------------------- Layout --------------------
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    with st.container(border=True):
        st.subheader("📌 Ma’lumotlarni kiriting")

        c1, c2 = st.columns(2)

        with c1:
            gender = st.selectbox("Jins (gender)", ["Female", "Male"])
            age = st.number_input("Yosh (age)", min_value=1, max_value=120, value=35)
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
            hba1c = st.number_input("HbA1c_level", min_value=3.0, max_value=15.0, value=5.5)

        with c2:
            smoking_history = st.selectbox(
                "Chekish tarixi (smoking_history)",
                ["never", "No Info", "current", "former", "ever", "not current"]
            )

            # ✅ UI’da "Bor/Yo‘q", lekin modelga 0/1 uzatiladi
            hypertension_label = st.selectbox("Gipertoniya (hypertension)", ["❌ Yo‘q", "✅ Bor"])
            heart_disease_label = st.selectbox("Yurak kasalligi (heart_disease)", ["❌ Yo‘q", "✅ Bor"])

            hypertension = 1 if hypertension_label.endswith("Bor") else 0
            heart_disease = 1 if heart_disease_label.endswith("Bor") else 0

            glucose = st.number_input("blood_glucose_level", min_value=50.0, max_value=400.0, value=110.0)

        st.write("")
        run = st.button("🔍 Baholash")

with right:
    with st.container(border=True):
        st.subheader("📊 Natija")

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

            risk_name, pill_class, reco_class, reco_title, reco_text = risk_bucket(proba)

            st.markdown(f"<p class='big-number'>{proba:.4f}</p>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge {pill_class}'>Risk: {risk_name}</span>", unsafe_allow_html=True)

            st.write("")
            st.progress(min(max(proba, 0.0), 1.0))

            st.markdown("<hr/>", unsafe_allow_html=True)

            st.subheader("📌 Yakuniy xulosa")
            if final_class == 1:
                st.error(f"🚨 Final klass: 1 (P ≥ {FINAL_THRESHOLD}) — juda yuqori ehtimol")
            else:
                st.success(f"✅ Final klass: 0 (P < {FINAL_THRESHOLD}) — kuzatish/skrining")

            st.markdown(f"""
            <div class="reco {reco_class}">
              <div class="reco-title">{reco_title}</div>
              <p class="reco-text">{reco_text}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("**Kiritilgan ma’lumotlar (tekshiruv):**")
            st.dataframe(input_df, use_container_width=True)

        else:
            st.info("Chap tomonda ma’lumotlarni kiriting va **Baholash** tugmasini bosing.")
            st.markdown("<div class='small-note'>Natija: ehtimollik (P), risk darajasi, final klass va tavsiya.</div>", unsafe_allow_html=True)

st.write("")
st.caption("© Diplom loyihasi: Tibbiy ma’lumotlarni AI orqali klassifikatsiya qilish (Clinical ML modul).")
