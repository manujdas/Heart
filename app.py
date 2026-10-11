"""
Heart Disease Detection - Responsive Streamlit App (v2)
Uses pre-trained RandomForest + scaler + feature_columns. No retraining.

Run:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Heart Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed",  # more room on phones
)

MODEL_DIR = Path(__file__).resolve().parent / "Model"
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# With get_dummies(drop_first=True) the alphabetically-first category is the
# dropped baseline. Display order below is only for a friendly dropdown.
DISPLAY_ORDER = {
    "cp": ["typical angina", "atypical angina", "non-anginal", "asymptomatic"],
    "restecg": ["normal", "st-t abnormality", "lv hypertrophy"],
    "slope": ["upsloping", "flat", "downsloping"],
    "thal": ["normal", "fixed defect", "reversable defect", "reversible defect"],
}

# ============================================================
# RESPONSIVE STYLING
# ============================================================
st.markdown(
    """
    <style>
    .block-container {padding: 1.5rem 1rem 3rem; max-width: 1200px;}
    .hero {
        background: linear-gradient(135deg, #b91c1c, #7f1d1d);
        color: #fff; padding: clamp(1rem, 3vw, 1.8rem) clamp(1rem, 4vw, 2rem);
        border-radius: 16px; margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; color: #fff; font-size: clamp(1.4rem, 4.5vw, 2.1rem);}
    .hero p  {margin: .35rem 0 0; opacity: .92; font-size: clamp(.85rem, 2.5vw, 1rem);}
    .result-box {
        border-radius: 16px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        color: #fff; text-align: center;
    }
    .result-box h2 {margin: 0; color: #fff; font-size: clamp(1.15rem, 4vw, 1.6rem);}
    .result-box p  {margin: .3rem 0 0; opacity: .95;}
    .risk-high {background: linear-gradient(135deg, #dc2626, #991b1b);}
    .risk-mod  {background: linear-gradient(135deg, #f59e0b, #b45309);}
    .risk-low  {background: linear-gradient(135deg, #16a34a, #166534);}
    .bar-wrap {width: 100%; height: 16px; border-radius: 10px;
               background: rgba(128,128,128,.25); overflow: hidden;}
    .bar-fill {height: 100%; border-radius: 10px;}
    div[data-testid="stMetricValue"] {font-size: clamp(1.2rem, 4vw, 1.8rem);}
    /* wrap columns instead of squeezing them on small screens */
    div[data-testid="stHorizontalBlock"] {flex-wrap: wrap; gap: .75rem;}
    div[data-testid="stColumn"] {min-width: min(100%, 220px);}
    @media (max-width: 640px) {
        .block-container {padding: 1rem .6rem 2rem;}
        div[data-testid="stColumn"] {min-width: 100%;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load(MODEL_DIR / "RF_heart.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_columns = list(joblib.load(MODEL_DIR / "feature_columns.pkl"))
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"❌ File not found: `{e.filename}`. Keep the `Model/` folder "
        "(RF_heart.pkl, scaler.pkl, feature_columns.pkl) next to app.py."
    )
    st.stop()
except Exception as e:
    st.error("❌ Could not load model files (often a scikit-learn version mismatch).")
    st.exception(e)
    st.stop()


BASELINE = {  # category dropped by get_dummies(drop_first=True) = alphabetically first
    "cp": "asymptomatic",
    "restecg": "lv hypertrophy",
    "slope": "downsloping",
    "thal": "fixed defect",
}


def build_category_options(feat: str):
    """Dropdown options = labels found in feature_columns + the dropped baseline."""
    trained = [c[len(feat) + 1:] for c in feature_columns if c.startswith(f"{feat}_")]
    ordered = [o for o in DISPLAY_ORDER[feat] if o in trained]
    ordered += [t for t in trained if t not in ordered]  # any unexpected spelling
    if BASELINE[feat] not in ordered:
        ordered.append(BASELINE[feat])
    return ordered


OPTIONS = {f: build_category_options(f) for f in CATEGORICAL_FEATURES}

# ============================================================
# PREPROCESS + PREDICT
# ============================================================
def preprocess(raw: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])

    scale_cols = list(
        getattr(scaler, "feature_names_in_", ["age", "trestbps", "chol", "thalch", "oldpeak"])
    )
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Manual one-hot (get_dummies(drop_first=True) breaks on a single row)
    for feat in CATEGORICAL_FEATURES:
        value = df.at[0, feat]
        df = df.drop(columns=[feat])
        col = f"{feat}_{value}"
        if col in feature_columns:
            df[col] = 1  # baseline -> all dummies 0

    return df.reindex(columns=feature_columns, fill_value=0).astype(float)


def predict(raw: dict):
    X = preprocess(raw)
    proba = model.predict_proba(X)[0]
    p_dis = float(proba[list(model.classes_).index(1)])
    return int(p_dis >= 0.5), p_dis, 1 - p_dis


def plausibility_warnings(v: dict):
    w = []
    exp_hr = 220 - v["age"]
    if v["thalch"] > exp_hr + 30:
        w.append(f"Max heart rate {v['thalch']} is far above expected (~{exp_hr}) for age {v['age']}.")
    if not 80 <= v["trestbps"] <= 200:
        w.append("Resting blood pressure is outside 80-200 mmHg.")
    if not 100 <= v["chol"] <= 600:
        w.append("Cholesterol is outside 100-600 mg/dL.")
    if v["oldpeak"] > 6:
        w.append("Oldpeak above 6 is extremely rare - please double-check.")
    return w


# ============================================================
# HEADER + SIDEBAR
# ============================================================
st.markdown(
    '<div class="hero"><h1>❤️ Heart Disease Detection</h1>'
    "<p>AI-assisted cardiovascular risk screening from clinical data.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("ℹ️ About")
    st.write("Estimates heart disease likelihood with a tuned **Random Forest** model.")
    with st.expander("📖 Feature guide"):
        st.markdown(
            "- **trestbps**: resting BP (mmHg)\n- **chol**: cholesterol (mg/dL)\n"
            "- **thalch**: max heart rate (bpm)\n- **oldpeak**: ST depression\n"
            "- **ca**: major vessels (0-3)\n- **cp**: chest pain type\n"
            "- **slope**: ST segment slope\n- **thal**: thalassemia result"
        )

# ============================================================
# FORM
# ============================================================
tab_input, tab_result = st.tabs(["🧑‍⚕️ Patient Input", "📊 Result"])

with tab_input:
    with st.form("patient_form", border=True):
        st.markdown("##### Demographics & Vitals")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 1, 120, 50)
        sex = c2.selectbox("Sex", ["Male", "Female"])
        c1, c2 = st.columns(2)
        trestbps = c1.number_input("Resting BP (mmHg)", 50, 250, 120)
        chol = c2.number_input("Cholesterol (mg/dL)", 50, 700, 200)
        c1, c2 = st.columns(2)
        thalch = c1.number_input("Max Heart Rate (bpm)", 50, 250, 150)
        fbs = c2.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])

        st.markdown("##### Exercise & Diagnostic Tests")
        c1, c2 = st.columns(2)
        exang = c1.selectbox("Exercise-Induced Angina", ["No", "Yes"])
        oldpeak = c2.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, 0.1)
        c1, c2 = st.columns(2)
        ca = c1.selectbox("Major Vessels (CA)", [0, 1, 2, 3])
        cp = c2.selectbox("Chest Pain Type", OPTIONS["cp"])
        c1, c2 = st.columns(2)
        restecg = c1.selectbox("Resting ECG", OPTIONS["restecg"])
        slope = c2.selectbox("ST Segment Slope", OPTIONS["slope"])
        thal = st.selectbox("Thalassemia", OPTIONS["thal"])

        submitted = st.form_submit_button(
            "🔍 Predict Heart Disease", use_container_width=True, type="primary"
        )

with tab_result:
    if not submitted:
        st.info("Fill in the patient form and press **Predict Heart Disease**.")
    else:
        values = {
            "age": age, "sex": 1 if sex == "Male" else 0, "trestbps": trestbps,
            "chol": chol, "fbs": 1 if fbs == "Yes" else 0, "thalch": thalch,
            "exang": 1 if exang == "Yes" else 0, "oldpeak": oldpeak, "ca": ca,
            "cp": cp, "restecg": restecg, "slope": slope, "thal": thal,
        }
        for w in plausibility_warnings(values):
            st.warning(f"⚠️ {w}")

        try:
            with st.spinner("Analysing..."):
                pred, p_dis, p_ok = predict(values)
        except Exception as e:
            st.error("❌ Prediction failed. Check inputs and model files.")
            st.exception(e)
        else:
            if 0.40 <= p_dis <= 0.60:
                css, color = "risk-mod", "#f59e0b"
                title = "⚠️ Heart Disease Detected" if pred else "✅ No Heart Disease Detected"
                sub = "Borderline result - clinical follow-up advised"
            elif pred == 1:
                css, color = "risk-high", "#dc2626"
                title, sub = "⚠️ Heart Disease Detected", "High likelihood of heart disease"
            else:
                css, color = "risk-low", "#16a34a"
                title, sub = "✅ No Heart Disease Detected", "Low likelihood of heart disease"

            st.markdown(
                f'<div class="result-box {css}"><h2>{title}</h2><p>{sub}</p></div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Disease Probability", f"{p_dis*100:.2f}%")
            m2.metric("No Disease Probability", f"{p_ok*100:.2f}%")
            st.markdown("**Disease risk level**")
            st.markdown(
                f'<div class="bar-wrap"><div class="bar-fill" '
                f'style="width:{p_dis*100:.1f}%;background:{color};"></div></div>',
                unsafe_allow_html=True,
            )
            with st.expander("🔎 Input summary"):
                st.dataframe(
                    pd.DataFrame({"Feature": list(values), "Value": [str(v) for v in values.values()]}),
                    hide_index=True, use_container_width=True,
                )

# ============================================================
# DISCLAIMER
# ============================================================
st.divider()
st.caption(
    "⚠️ **Medical Disclaimer:** For educational and research purposes only. Not a "
    "medical device and not a substitute for professional diagnosis, advice, or "
    "treatment. Always consult a qualified healthcare provider."
)