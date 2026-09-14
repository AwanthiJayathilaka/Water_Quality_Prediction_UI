# ============================================================
# app.py
# HydroLab - Smart Water Quality Prediction Interface
# ============================================================

import os
import json
import base64
import joblib
import pandas as pd
import streamlit as st


# ============================================================
# 1. File paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

LOGO_PATH = os.path.join(BASE_DIR, "assets", "hydrolab_logo.png")
METRICS_FILE = os.path.join(BASE_DIR, "saved_models", "model_metrics.json")


# ============================================================
# 2. Page settings
# ============================================================

st.set_page_config(
    page_title="HydroLab",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "💧",
    layout="centered"
)


# ============================================================
# 3. Convert image to base64
# ============================================================

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded


# ============================================================
# 4. Custom CSS design
# ============================================================

st.markdown(
    """
    <style>

    /* Main page width and spacing */
    .block-container {
        padding-top: 45px;
        max-width: 1000px;
    }

    /* Hide Streamlit menu and footer */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* HydroLab logo card */
    .hydrolab-hero {
        background: linear-gradient(135deg, rgba(0, 90, 140, 0.18), rgba(0, 170, 130, 0.10));
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 30px;
        padding: 55px 35px;
        margin-top: 20px;
        margin-bottom: 55px;
        box-shadow: 0 20px 55px rgba(0, 0, 0, 0.32);
        text-align: center;
    }

    .hydrolab-logo {
        width: 430px;
        max-width: 90%;
        height: auto;
        display: block;
        margin-left: auto;
        margin-right: auto;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    .hydrolab-subtitle {
        font-size: 20px;
        line-height: 1.7;
        max-width: 850px;
        margin-top: 28px;
        margin-left: auto;
        margin-right: auto;
        color: rgba(255, 255, 255, 0.88);
    }

    /* Section headings */
    h2 {
        font-size: 42px !important;
        font-weight: 800 !important;
        margin-top: 45px !important;
        margin-bottom: 20px !important;
    }

    h3 {
        font-size: 28px !important;
        font-weight: 750 !important;
    }

    p, label, div {
        font-size: 18px;
    }

    /* Button design */
    .stButton > button {
        font-size: 18px;
        font-weight: 750;
        padding: 12px 32px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.20);
    }

    .stButton > button:hover {
        border: 1px solid rgba(0, 200, 255, 0.80);
        transform: scale(1.01);
    }

    .stNumberInput input {
        font-size: 18px;
    }

    /* Result card */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 20px;
        padding: 24px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    /* Small note card */
    .small-note {
        background: rgba(0, 150, 255, 0.10);
        border-left: 5px solid rgba(0, 180, 255, 0.80);
        padding: 16px 18px;
        border-radius: 12px;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. HydroLab logo header
# ============================================================

if os.path.exists(LOGO_PATH):
    logo_base64 = image_to_base64(LOGO_PATH)

    st.markdown(
        f"""
        <div class="hydrolab-hero">
            <img class="hydrolab-logo" src="data:image/png;base64,{logo_base64}">
            <div class="hydrolab-subtitle">
                A machine learning-based water quality prediction interface for estimating
                selected aquarium water quality parameters using trained prediction models.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    st.error("Logo image not found. Please place your logo inside assets/hydrolab_logo.png")


# ============================================================
# 6. Load saved model details
# ============================================================

if not os.path.exists(METRICS_FILE):
    st.error("Model details not found. Please run train_models.py first.")
    st.stop()

with open(METRICS_FILE, "r") as f:
    model_metrics = json.load(f)


# ============================================================
# 7. Prediction options
# ============================================================

prediction_options = [
    "ORP (mV)",
    "NH4+ (ppm)",
    "NO3- (ppm)",
    "Chlo. a (ppb)",
    "Phycocyanin (ppb)",
    "Sal (psu)"
]

available_prediction_options = [
    option for option in prediction_options if option in model_metrics
]

if len(available_prediction_options) == 0:
    st.error("No trained models are available. Please run train_models.py again.")
    st.stop()


# ============================================================
# 8. Find required real input values
# ============================================================

def get_required_raw_inputs(target, visited=None):
    if visited is None:
        visited = set()

    if target in visited:
        return []

    visited.add(target)

    raw_inputs = []
    features = model_metrics[target]["features"]

    for feature in features:

        if feature in model_metrics and feature != target:
            sub_inputs = get_required_raw_inputs(feature, visited)

            for item in sub_inputs:
                if item not in raw_inputs:
                    raw_inputs.append(item)

        else:
            if feature not in raw_inputs:
                raw_inputs.append(feature)

    return raw_inputs


# ============================================================
# 9. Decide prediction order
# ============================================================

def get_prediction_order(target, visited=None):
    if visited is None:
        visited = set()

    order = []

    if target in visited:
        return order

    visited.add(target)

    features = model_metrics[target]["features"]

    for feature in features:

        if feature in model_metrics and feature != target:
            sub_order = get_prediction_order(feature, visited)

            for item in sub_order:
                if item not in order:
                    order.append(item)

    if target not in order:
        order.append(target)

    return order


# ============================================================
# 10. Predict one parameter
# ============================================================

def predict_parameter(target, available_values):
    model_file = model_metrics[target]["model_file"]

    if not os.path.isabs(model_file):
        model_file = os.path.join(BASE_DIR, model_file)

    if not os.path.exists(model_file):
        st.error(f"Model file not found for {target}")
        return None

    saved_object = joblib.load(model_file)

    model = saved_object["model"]
    features = saved_object["features"]

    missing_features = []

    for feature in features:
        if feature not in available_values:
            missing_features.append(feature)

    if len(missing_features) > 0:
        st.error(
            f"Cannot predict {target}. Missing required values: "
            + ", ".join(missing_features)
        )
        return None

    input_data = pd.DataFrame(
        [[available_values[feature] for feature in features]],
        columns=features
    )

    predicted_value = model.predict(input_data)[0]

    return predicted_value


# ============================================================
# 11. Step 1: Ask user what they want to predict
# ============================================================

st.header("Step 1: What do you want to predict?")

st.markdown(
    """
    <div class="small-note">
        Select one or more water quality parameters. If your selected parameter needs another
        parameter first, HydroLab will predict that required parameter automatically.
    </div>
    """,
    unsafe_allow_html=True
)

selected_predictions = st.multiselect(
    "Select prediction parameter(s):",
    available_prediction_options
)

continue_button = st.button("Continue")


# ============================================================
# 12. Store selected predictions
# ============================================================

if "selected_predictions" not in st.session_state:
    st.session_state.selected_predictions = []

if continue_button:

    if len(selected_predictions) == 0:
        st.warning("Please select at least one parameter to predict.")

    else:
        st.session_state.selected_predictions = selected_predictions


# ============================================================
# 13. Step 2: Explain required parameters
# ============================================================

if len(st.session_state.selected_predictions) > 0:

    selected_predictions = st.session_state.selected_predictions

    st.header("Step 2: HydroLab checks what is needed")

    all_prediction_order = []
    all_raw_inputs = []

    for target in selected_predictions:

        prediction_order = get_prediction_order(target)
        raw_inputs = get_required_raw_inputs(target)

        for item in prediction_order:
            if item not in all_prediction_order:
                all_prediction_order.append(item)

        for item in raw_inputs:
            if item not in all_raw_inputs:
                all_raw_inputs.append(item)

        st.subheader(f"For predicting {target}")

        if len(prediction_order) > 1:

            intermediate_predictions = prediction_order[:-1]

            st.info(
                f"To predict {target}, HydroLab first needs to predict "
                + ", ".join(intermediate_predictions)
                + ". You do not need to enter these values manually."
            )

            st.write(
                "HydroLab will first calculate "
                + ", ".join(intermediate_predictions)
                + f", and then it will calculate {target}."
            )

        else:

            st.info(
                f"{target} can be predicted directly using the required input values."
            )

        st.write(
            "You need to enter: **"
            + ", ".join(raw_inputs)
            + "**"
        )


    # ============================================================
    # 14. Step 3: Input values
    # ============================================================

    st.header("Step 3: Enter required input values")

    st.write(
        "Please enter only the values shown below. "
        "HydroLab will calculate intermediate predicted values automatically."
    )

    input_values = {}

    for parameter in all_raw_inputs:

        input_values[parameter] = st.number_input(
            f"Enter {parameter}",
            value=0.0,
            step=0.01
        )


    # ============================================================
    # 15. Step 4: Predict
    # ============================================================

    st.header("Step 4: Predict")

    predict_button = st.button("🔍 Predict Now")

    if predict_button:

        st.subheader("Prediction Process")

        available_values = input_values.copy()
        prediction_results = {}

        for target in all_prediction_order:

            st.write(f"Predicting {target}...")

            predicted_value = predict_parameter(target, available_values)

            if predicted_value is None:
                st.error(f"{target} prediction failed.")
                continue

            available_values[target] = predicted_value
            prediction_results[target] = predicted_value

            st.success(f"{target} predicted successfully.")

        st.header("Final Prediction Results")

        for target in all_prediction_order:

            if target not in prediction_results:
                continue

            predicted_value = prediction_results[target]

            r2 = model_metrics[target]["r2"]
            mae = model_metrics[target]["mae"]
            rmse = model_metrics[target]["rmse"]
            accuracy = model_metrics[target]["accuracy_percent"]

            if target in selected_predictions:
                result_type = "Required Prediction"
            else:
                result_type = "Intermediate Prediction"

            st.markdown(
                f"""
                <div class="result-card">
                    <h3>{result_type}: {target}</h3>
                    <p><b>Predicted {target}:</b> {predicted_value:.4f}</p>
                    <p><b>Model Accuracy Based on R²:</b> {accuracy:.2f}%</p>
                    <p><b>R² Score:</b> {r2:.4f}</p>
                    <p><b>MAE:</b> {mae:.4f}</p>
                    <p><b>RMSE:</b> {rmse:.4f}</p>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# 16. Simple model explanation
# ============================================================

st.header("Simple Model Explanation")

st.write(
    "HydroLab uses trained machine learning models to predict water quality parameters."
)

st.write("**ORP (mV):** predicted using pH, Temp (°C), and TDS (ppm).")

st.write(
    "**NH4+ (ppm):** needs ORP (mV). "
    "If ORP is not entered by the user, HydroLab predicts ORP first."
)

st.write("**NO3- (ppm):** predicted using pH and TDS (ppm).")

st.write(
    "**Chlo. a (ppb):** needs Salinity. "
    "If Salinity is not entered by the user, HydroLab predicts Salinity first."
)

st.write("**Phycocyanin (ppb):** predicted using Temp (°C) and TDS (ppm).")

st.write("**Sal (psu):** predicted using EC, TDS, and Turbidity.")


# ============================================================
# 17. Footer
# ============================================================

st.divider()

st.write(
    "Developed as a machine learning-based user interface for aquarium water quality prediction."
)