"""
Streamlit UI for the churn predictor.

Run with:  streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st

from src.predict import ChurnPredictor

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


@st.cache_resource
def get_predictor() -> ChurnPredictor:
    return ChurnPredictor()


def predict_page():
    st.title("📉 Customer Churn Prediction")
    st.caption("Artificial Neural Network trained on bank customer data")

    try:
        predictor = get_predictor()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python -m src.train` first to "
            "generate the model and preprocessing artifacts."
        )
        st.stop()

    geo_options = list(predictor.geo_encoder.categories_[0])
    gender_options = list(predictor.gender_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        credit_score = st.slider("Credit Score", 300, 900, 650)
        geography = st.selectbox("Geography", geo_options)
        gender = st.selectbox("Gender", gender_options)
        age = st.slider("Age", 18, 92, 38)
        tenure = st.slider("Tenure (years)", 0, 10, 5)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0, step=1000.0)
        num_products = st.slider("Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=50000.0, step=1000.0
        )

    if st.button("Predict Churn", type="primary"):
        customer = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
        }
        result = predictor.predict(customer)
        prob = result["churn_probability"]

        st.metric("Churn Probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        if result["will_churn"]:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is likely to stay.")


def comparison_page():
    st.title("📊 Model Comparison")
    st.caption("ANN vs. classical ML classifiers, evaluated on the identical held-out test set")

    csv_path = "artifacts/model_comparison.csv"
    png_path = "artifacts/model_comparison.png"

    if not os.path.exists(csv_path):
        st.warning(
            "No comparison results found yet. Run `python -m src.compare_models` "
            "in your terminal first (after training), then reload this page."
        )
        st.stop()

    df = pd.read_csv(csv_path)
    st.dataframe(
        df.style.highlight_max(
            subset=["accuracy", "precision", "recall", "f1", "roc_auc"], color="#2f5d3a"
        ),
        use_container_width=True,
    )

    best_model = df.sort_values("roc_auc", ascending=False).iloc[0]
    st.info(f"🏆 Best by ROC-AUC: **{best_model['model']}** ({best_model['roc_auc']:.4f})")

    if os.path.exists(png_path):
        st.image(png_path, use_container_width=True)


def monitoring_page():
    st.title("🩺 Monitoring & Drift Detection")
    st.caption("Tracks every prediction the app makes and checks for data drift vs. training data")

    log_path = "artifacts/prediction_log.csv"
    if not os.path.exists(log_path):
        st.info("No predictions logged yet. Make a prediction on the Predict page first.")
        return

    logs = pd.read_csv(log_path)
    st.metric("Total predictions logged", len(logs))
    st.metric("Predicted churn rate", f"{(logs['churn_probability'] >= 0.5).mean():.1%}")

    st.subheader("Churn probability distribution")
    if len(logs) >= 2:
       bin_counts = pd.cut(logs["churn_probability"], bins=10).value_counts().sort_index()
       bin_counts.index = bin_counts.index.astype(str)
       st.bar_chart(bin_counts)
    else:
        st.info("Log a few more predictions to see the distribution chart.")

    st.subheader("Recent predictions")
    st.dataframe(logs.tail(20), use_container_width=True)

    st.subheader("Data drift check")
    st.caption(
        "Compares the distribution of inputs the app has actually received "
        "against the training data distribution, using a KS test per feature."
    )
    if st.button("Run drift check"):
        from src.monitoring import check_drift

        result = check_drift()
        if "error" in result:
            st.warning(result["error"])
        else:
            if result["any_drift_detected"]:
                st.error("⚠️ Drift detected in at least one feature — consider retraining.")
            else:
                st.success("✅ No significant drift detected.")
            st.json(result)


def main():
    page = st.sidebar.radio("Navigate", ["Predict", "Model Comparison", "Monitoring"])
    if page == "Predict":
        predict_page()
    elif page == "Model Comparison":
        comparison_page()
    else:
        monitoring_page()


if __name__ == "__main__":
    main()
