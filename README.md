# Fraud Detection

A logistic regression classifier that scores mobile-money transactions for fraud, trained on 6.4 million simulated transfers from the PaySim dataset. A Streamlit app wraps the model so you can score a transaction by hand and read how the model and its data were built.

## Results

Measured on a held-out test set of 1,899,033 transactions containing 4,570 fraud cases.

| Metric | Value | Meaning |
|---|---|---|
| PR-AUC | 0.674 | The baseline for this dataset is 0.0024, so the model scores 280x higher. |
| Precision | 0.771 | Of the transactions the model flags, 77% are fraud. |
| Recall | 0.539 | The model catches 54% of real fraud and misses the rest. |
| F1 | 0.635 | At the tuned threshold of 0.9926. |

Accuracy is left out on purpose. At a 0.13% fraud rate, a model that labels everything legitimate already scores 99.87%.

## The dataset

PaySim generates synthetic mobile-money traffic by modelling an African payment network and injecting fraudulent agents into it. Banks cannot release real transaction data, so researchers train on this instead.

| | |
|---|---|
| Transactions | 6,362,620 |
| Fraud cases | 8,213 (0.13%, one in every 774) |
| Time span | 743 hourly steps, about 31 days |
| Columns | 11 |

Fraud appears in two of the five transaction types. Fraudsters transfer a victim's balance out, then cash it.

| Type | Transactions | Fraud | Fraud rate |
|---|---|---|---|
| CASH_OUT | 2,237,500 | 4,116 | 0.184% |
| PAYMENT | 2,151,495 | 0 | 0% |
| CASH_IN | 1,399,284 | 0 | 0% |
| TRANSFER | 532,909 | 4,097 | 0.769% |
| DEBIT | 41,432 | 0 | 0% |

### Columns

| Column | Meaning | Role |
|---|---|---|
| `step` | Hour of the simulation, 1 to 743 | Split only |
| `type` | Transaction type, one of five categories | Feature |
| `amount` | Transaction amount | Feature |
| `nameOrig` | Sender account id | Dropped |
| `oldbalanceOrg` | Sender balance before the transaction | Feature |
| `newbalanceOrig` | Sender balance after the transaction | Feature |
| `nameDest` | Recipient account id | Dropped |
| `oldbalanceDest` | Recipient balance before the transaction | Feature |
| `newbalanceDest` | Recipient balance after the transaction | Feature |
| `isFraud` | Whether the transaction was fraudulent | Target |
| `isFlaggedFraud` | Flag raised by the simulator's own rule | Dropped |

Three columns stay out of the model. `nameOrig` and `nameDest` hold account ids, and senders are almost all unique, so the model would memorise identifiers instead of learning behaviour. `isFlaggedFraud` holds the simulator's own fraud flag, which leaks the answer into a fraud model and fires on 16 of 6.36 million rows anyway. `step` marks the train/test boundary, and the `ColumnTransformer` leaves it out of the features.

## How the model works

One scikit-learn `Pipeline` holds every step, so each transformer fits on the training data alone and ships inside the same pickle file:

1. `StandardScaler` scales the five numeric columns.
2. `OneHotEncoder(drop="first")` encodes the transaction type.
3. `LogisticRegression(class_weight="balanced")` reweights the rare class instead of resampling it.

### Splitting by time

PaySim records each fraud as a pair of rows, a `TRANSFER` followed by a `CASH_OUT` sharing the same hour and amount, with identical sender balances in 98.4% of cases. A random split sent 1,665 of 4,080 pairs to opposite sides, so the model trained on near-copies of the rows it was later tested on. The fraud rate also moves by a factor of 4.8 across time quartiles, which a random split hides.

Cutting the data at hour 323, the 70th percentile of `step`, keeps each pair on one side and stops future transactions from reaching the training set.

### Tuning the threshold

`predict()` decides at 0.5, which flags about 5% of legitimate transactions. Reading the threshold off the precision-recall curve instead puts it at 0.9926. That change raised F1 from 0.04 to 0.635 and cut false positives from 101,499 to 734.

## Where it falls short

- **The model misses about 46% of fraud.** Lowering the threshold catches more fraud and flags more innocent customers.
- **Logistic regression draws a straight line.** Fraud here follows a rule, drain the account then cash out. Swapping in `HistGradientBoostingClassifier` measures at PR-AUC 0.955, precision 0.937 and recall 0.820 on the same split, so the model family is the ceiling here rather than the features.
- **A simulator produced the data.** Precision and recall on real transactions would differ.
- **The model has no account history.** The features exclude sender and recipient ids, so it cannot tell that an account behaved oddly last week.
- **It keeps extrapolating past what it saw.** A transfer of 10^15 scores 1.0000. The features are scaled linearly, so values far outside the training range push the score further rather than saturating. Fed uniformly random inputs, the model calls 19% of them fraud, including transaction types that carry no fraud at all in the training data.

## Running it

Requires Python 3.13.

```bash
git clone https://github.com/Gio71220924/Fraud-Detection.git
cd Fraud-Detection

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS and Linux

pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

Then launch the app:

```bash
streamlit run fraud_detection.py
```

It opens at http://localhost:8501 and offers an English or Bahasa Indonesia interface.

### Retraining

`fraud_detection_model.pkl` is committed, so the app runs without the dataset. Retraining needs it. Download the PaySim synthetic financial dataset, save it at the repository root as `AIML Dataset.csv`, and run `model.ipynb` from the top. The CSV is 470 MB and stays out of version control.

Point the notebook kernel at `.venv`. The pickle must come from the same environment that serves it: a model written by scikit-learn 1.5.2 cannot be unpickled by 1.9.0.

## Files

| File | Purpose |
|---|---|
| `model.ipynb` | EDA, feature selection, training, threshold tuning, model export |
| `fraud_detection.py` | Streamlit app |
| `fraud_detection_model.pkl` | The fitted pipeline |

## Built with

Python 3.13, scikit-learn 1.9.0, pandas 3.0.5, Streamlit 1.62.0, joblib 1.5.3, matplotlib 3.11.1, seaborn 0.13.2.
