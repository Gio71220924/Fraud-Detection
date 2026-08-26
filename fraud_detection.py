import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "fraud_detection_model.pkl"

# ponytail: threshold hasil tuning PR-curve di model.ipynb (F1 terbaik).
# Default predict() memakai 0.5, yang menandai ~5% transaksi sah sebagai fraud.
# Tune ulang dan perbarui angka ini setiap model dilatih ulang; pindahkan ke
# dalam .pkl kalau ternyata sering berubah.
THRESHOLD = 0.9926

# Label yang dilihat user -> kategori persis seperti saat model dilatih.
# Model dilatih pada nilai kolom `type` di dataset: PAYMENT, TRANSFER,
# CASH_OUT, CASH_IN, DEBIT. Kirim "Payment" mentah -> OneHotEncoder error.
TYPES = {
    "Payment": "PAYMENT",
    "Transfer": "TRANSFER",
    "Cash Out": "CASH_OUT",
    "Cash In": "CASH_IN",
    "Debit": "DEBIT",
}

model = joblib.load(MODEL_PATH)

trained = set(model.named_steps["preprocessor"].named_transformers_["cat"].categories_[0])
assert set(TYPES.values()) <= trained, f"kategori tak dikenal model: {set(TYPES.values()) - trained}"

st.title("Fraud Detection Model")

st.markdown("This app predicts whether a transaction is fraudulent or not based on the input features.")
st.markdown("Please enter the transaction details below:")

st.divider()

transaction_type = st.selectbox("Transaction Type", list(TYPES))
amount = st.number_input("Transaction Amount", min_value=0.0, step=0.01, value = 1000.0)
oldbalanceorg = st.number_input("Old Balance of Origin Account", min_value=0.0, step=0.01, value = 1000.0)
newbalanceorg = st.number_input("New Balance of Origin Account", min_value=0.0, step=0.01, value = 1000.0)
oldbalancedest = st.number_input("Old Balance of Destination Account", min_value=0.0, step=0.01, value = 1000.0)
newbalancedest = st.number_input("New Balance of Destination Account", min_value=0.0, step=0.01, value = 1000.0)


if st.button("Predict"):
    input_data = pd.DataFrame({
        "type": [TYPES[transaction_type]],
        "amount": [amount],
        "oldbalanceOrg": [oldbalanceorg],
        "newbalanceOrig": [newbalanceorg],
        "oldbalanceDest": [oldbalancedest],
        "newbalanceDest": [newbalancedest]
    })

    fraud_proba = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if fraud_proba >= THRESHOLD:
        st.error(f"The transaction is predicted to be **fraudulent** (fraud probability {fraud_proba:.4f}, threshold {THRESHOLD}).")
    else:
        st.success(f"The transaction is predicted to be **legitimate** (fraud probability {fraud_proba:.4f}, threshold {THRESHOLD}).")

    st.caption(f"Tuned threshold {THRESHOLD} from the PR curve. At the default 0.5 this model produces ~120x more false positives.")
