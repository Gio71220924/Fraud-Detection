import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud detection",
    page_icon=":material/shield_person:",
    layout="centered",
)

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
    "Cash out": "CASH_OUT",
    "Cash in": "CASH_IN",
    "Debit": "DEBIT",
}

# ponytail: angka di bawah di-hardcode dari model.ipynb. Dataset mentahnya
# 470 MB — app tidak boleh membacanya hanya untuk menampilkan statistik.
TYPE_STATS = pd.DataFrame(
    [
        ("CASH_OUT", 2_237_500, 4_116, 0.184),
        ("PAYMENT", 2_151_495, 0, 0.000),
        ("CASH_IN", 1_399_284, 0, 0.000),
        ("TRANSFER", 532_909, 4_097, 0.769),
        ("DEBIT", 41_432, 0, 0.000),
    ],
    columns=["Type", "Transactions", "Fraud", "Fraud rate (%)"],
)

COLUMNS = pd.DataFrame(
    [
        ("step", "Hour of the simulation, 1 to 743 (31 days)", "Split only"),
        ("type", "Transaction type, one of five categories", "Feature"),
        ("amount", "Transaction amount", "Feature"),
        ("nameOrig", "Sender account id", "Dropped"),
        ("oldbalanceOrg", "Sender balance before the transaction", "Feature"),
        ("newbalanceOrig", "Sender balance after the transaction", "Feature"),
        ("nameDest", "Recipient account id", "Dropped"),
        ("oldbalanceDest", "Recipient balance before the transaction", "Feature"),
        ("newbalanceDest", "Recipient balance after the transaction", "Feature"),
        ("isFraud", "Whether the transaction was fraudulent", "Target"),
        ("isFlaggedFraud", "Flag raised by the simulator's own rule", "Dropped"),
    ],
    columns=["Column", "Meaning", "Role"],
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

trained = set(model.named_steps["preprocessor"].named_transformers_["cat"].categories_[0])
assert set(TYPES.values()) <= trained, f"kategori tak dikenal model: {set(TYPES.values()) - trained}"

with st.sidebar:
    st.subheader("Model card")
    st.metric("PR-AUC", "0.674", help="Baseline for this dataset is 0.0024, so this is a 280x lift.")
    st.metric("Precision", "0.771", help="Of the transactions flagged as fraud, 77% really are.")
    st.metric("Recall", "0.539", help="Of all real fraud, 54% gets caught. The rest slips through.")
    st.caption(f"Logistic regression, decision threshold {THRESHOLD}.")
    st.caption("Trained on PaySim, a synthetic dataset. Not a production fraud system.")

st.title("Fraud detection model")
st.caption(
    "Logistic regression trained on 6.4 million simulated mobile-money transfers. "
    "Enter a transaction below to score it, or read how the model and its data were built."
)

predict_tab, project_tab, data_tab = st.tabs(["Predict", "The project", "The dataset"])

with predict_tab:
    with st.form("transaction"):
        transaction_type = st.selectbox("Transaction type", list(TYPES))
        amount = st.number_input("Transaction amount", min_value=0.0, step=0.01, value=1000.0)

        left, right = st.columns(2)
        with left:
            st.markdown("**Origin account**")
            oldbalanceorg = st.number_input("Balance before", min_value=0.0, step=0.01, value=1000.0)
            newbalanceorg = st.number_input("Balance after", min_value=0.0, step=0.01, value=1000.0)
        with right:
            st.markdown("**Destination account**")
            oldbalancedest = st.number_input(
                "Balance before", min_value=0.0, step=0.01, value=1000.0, key="dest_old"
            )
            newbalancedest = st.number_input(
                "Balance after", min_value=0.0, step=0.01, value=1000.0, key="dest_new"
            )

        submitted = st.form_submit_button("Predict", icon=":material/play_arrow:", type="primary")

    if submitted:
        input_data = pd.DataFrame(
            {
                "type": [TYPES[transaction_type]],
                "amount": [amount],
                "oldbalanceOrg": [oldbalanceorg],
                "newbalanceOrig": [newbalanceorg],
                "oldbalanceDest": [oldbalancedest],
                "newbalanceDest": [newbalancedest],
            }
        )

        fraud_proba = model.predict_proba(input_data)[0][1]

        st.subheader("Result")
        if fraud_proba >= THRESHOLD:
            st.error(
                f"Predicted **fraudulent** — fraud probability {fraud_proba:.4f}, "
                f"at or above the {THRESHOLD} threshold.",
                icon=":material/gpp_bad:",
            )
        else:
            st.success(
                f"Predicted **legitimate** — fraud probability {fraud_proba:.4f}, "
                f"below the {THRESHOLD} threshold.",
                icon=":material/verified_user:",
            )

        st.progress(min(fraud_proba, 1.0), text=f"Fraud probability {fraud_proba:.1%}")

        if TYPES[transaction_type] not in ("TRANSFER", "CASH_OUT"):
            st.caption(
                f"Note: the training data contains zero fraud of type {TYPES[transaction_type]}, "
                "so this type is predicted legitimate almost regardless of the amounts."
            )

with project_tab:
    st.subheader("What this is")
    st.markdown(
        "A binary classifier that estimates whether a single mobile-money transaction is "
        "fraudulent. It is a portfolio and teaching project built on public synthetic data, "
        "not a system connected to any real payment rail."
    )

    st.subheader("How the model is built")
    st.markdown(
        """
        A single scikit-learn `Pipeline`, so every preprocessing step is fitted on the
        training data alone and travels with the model into this app:

        1. `StandardScaler` on the five numeric columns.
        2. `OneHotEncoder(drop="first")` on the transaction type.
        3. `LogisticRegression(class_weight="balanced")`, which reweights the rare class
           instead of resampling it.
        """
    )

    st.subheader("Two decisions that matter more than the algorithm")
    with st.container(border=True):
        st.markdown("**The split is by time, not random**")
        st.markdown(
            "PaySim records each fraud as a pair of rows — a `TRANSFER` followed by a "
            "`CASH_OUT` sharing the same hour and amount, with identical sender balances in "
            "98.4% of cases. A random split sent 1,665 of 4,080 such pairs to opposite sides, "
            "letting the model meet a near-copy of each test row during training. Splitting at "
            "hour 323 keeps every pair on one side and stops future transactions leaking "
            "backwards into training."
        )
    with st.container(border=True):
        st.markdown("**Accuracy is not the metric**")
        st.markdown(
            "At a 0.13% fraud rate, predicting *legitimate* for everything scores 99.87% "
            "accuracy. The model is scored on PR-AUC instead, and its decision threshold is "
            f"tuned on the precision-recall curve to {THRESHOLD} rather than left at the "
            "default 0.5. That single change moved F1 from 0.04 to 0.635 and cut false "
            "positives from 101,499 to 734."
        )

    st.subheader("Where it falls short")
    st.markdown(
        f"""
        - **It misses about 46% of fraud.** Recall is 0.539 at this threshold. Lowering
          `THRESHOLD` below {THRESHOLD} catches more fraud and flags more innocent customers.
        - **Logistic regression is a linear model.** Fraud here is a rule-like pattern
          (drain the account, then cash out), which gradient boosting fits better.
        - **The data is simulated.** PaySim comes out of a simulator, so real-world precision
          and recall would differ.
        - **No account history.** Sender and recipient ids are dropped, so the model cannot
          see that an account has behaved oddly before.
        """
    )

with data_tab:
    st.subheader("PaySim mobile-money transactions")
    st.markdown(
        "A synthetic dataset generated by the PaySim simulator, which models mobile-money "
        "traffic on an African payment network and injects fraudulent agents into it. It is "
        "public precisely because real transaction data cannot be released."
    )

    cols = st.columns(4)
    cols[0].metric("Transactions", "6.36M")
    cols[1].metric("Fraud cases", "8,213")
    cols[2].metric("Fraud rate", "0.13%")
    cols[3].metric("Time span", "31 days")

    st.markdown(
        "That works out to roughly **one fraud in every 774 transactions** — the imbalance "
        "that drives nearly every modelling decision in this project."
    )

    st.subheader("Transaction types")
    st.dataframe(TYPE_STATS, hide_index=True, width="stretch")
    st.markdown(
        "Fraud appears in only two of the five types. `TRANSFER` and `CASH_OUT` carry every "
        "fraudulent record; `PAYMENT`, `CASH_IN` and `DEBIT` contain none at all. The pattern "
        "is a transfer that empties an account, immediately followed by a cash-out."
    )

    st.subheader("Columns")
    st.dataframe(COLUMNS, hide_index=True, width="stretch")

    with st.expander("Why three columns are dropped", icon=":material/delete:"):
        st.markdown(
            """
            - `nameOrig` and `nameDest` are account ids. Senders are almost all unique, so the
              model would memorise identifiers rather than learn behaviour.
            - `isFlaggedFraud` is the simulator's own fraud flag. Feeding a fraud label into a
              fraud model is leakage, and it fires on only 16 of 6.36 million rows anyway.
            - `step` stays in the table but never reaches the model. It defines the train/test
              boundary, and the `ColumnTransformer` drops it from the features.
            """
        )

    st.caption(
        "Fraudulent transactions are also much larger: a median of 441,424 against 74,872 "
        "across all transactions."
    )
