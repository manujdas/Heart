import joblib
import pandas as pd


# ============================================================
# 1. LOAD SAVED MODEL, SCALER AND FEATURE COLUMNS
# ============================================================

model = joblib.load("Model/RF_heart.pkl")
scaler = joblib.load("Model/scaler.pkl")
feature_columns = joblib.load("Model/feature_columns.pkl")

print("Model loaded successfully!")
print("Scaler loaded successfully!")
print("Feature columns loaded successfully!")

print("\nNumber of model features:", len(feature_columns))
print("Features:")
print(feature_columns)


# ============================================================
# 2. PREDICTION FUNCTION
# ============================================================

def predict_heart_disease(
    age,
    sex,
    trestbps,
    chol,
    fbs,
    thalch,
    exang,
    oldpeak,
    ca,
    cp,
    restecg,
    slope,
    thal
):

    # --------------------------------------------------------
    # STEP 1: Create patient DataFrame
    # --------------------------------------------------------

    patient = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "thalch": [thalch],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "ca": [ca],
        "cp": [cp],
        "restecg": [restecg],
        "slope": [slope],
        "thal": [thal]
    })


    # --------------------------------------------------------
    # STEP 2: Scale ONLY numerical features
    # --------------------------------------------------------

    numerical_features = [
        "age",
        "trestbps",
        "chol",
        "thalch",
        "oldpeak"
    ]

    patient[numerical_features] = scaler.transform(
        patient[numerical_features]
    )


    # --------------------------------------------------------
    # STEP 3: One-hot encode categorical features
    # --------------------------------------------------------

    categorical_features = [
        "cp",
        "restecg",
        "slope",
        "thal"
    ]

    patient = pd.get_dummies(
        patient,
        columns=categorical_features,
        drop_first=True
    )


    # --------------------------------------------------------
    # STEP 4: Make sure all training columns exist
    # --------------------------------------------------------

    patient = patient.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # STEP 5: Convert to numeric
    # --------------------------------------------------------

    patient = patient.astype(float)


    # --------------------------------------------------------
    # STEP 6: Prediction
    # --------------------------------------------------------

    prediction = model.predict(patient)[0]


    # --------------------------------------------------------
    # STEP 7: Probability
    # --------------------------------------------------------

    probability = model.predict_proba(patient)[0]


    # --------------------------------------------------------
    # STEP 8: Result
    # --------------------------------------------------------

    if prediction == 1:

        result = "Heart Disease Detected"

    else:

        result = "No Heart Disease Detected"


    return result, probability


# ============================================================
# 3. TEST PATIENT
# ============================================================

result, probability = predict_heart_disease(

    age=55,

    sex=1,

    trestbps=140,

    chol=250,

    fbs=0,

    thalch=150,

    exang=0,

    oldpeak=1.2,

    ca=0,

    # Chest pain
    cp="non-anginal",

    # Resting ECG
    restecg="st-t abnormality",

    # Slope
    slope="flat",

    # Thal
    thal="reversible defect"
)


# ============================================================
# 4. DISPLAY RESULT
# ============================================================

print("\n==========================================")
print("       HEART DISEASE PREDICTION")
print("==========================================")

print("Prediction:", result)

print(
    "No Disease Probability:",
    round(probability[0] * 100, 2),
    "%"
)

print(
    "Disease Probability:",
    round(probability[1] * 100, 2),
    "%"
)

print("==========================================")