import altair as alt
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Fraud detection",
    page_icon=":material/shield_person:",
    layout="centered",
)

MODEL_PATH = "fraud_detection_model.pkl"
CURVE_PATH = "pr_curve.csv"
LEARNING_PATH = "learning_curve.csv"
EXAMPLES_PATH = "examples.csv"

# Palet kategorikal tervalidasi. Slot 1 untuk kurva, slot 2 untuk titik operasi.
# Dua-duanya lolos enam pemeriksaan (lightness band, chroma, pemisahan CVD,
# normal-vision floor, kontras) di light dan dark, jangan diganti tanpa
# menjalankan ulang validator.
PALETTE = {
    "light": {"series": "#2a78d6", "accent": "#eb6834", "surface": "#fcfcfb",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "muted": "#898781", "ink": "#0b0b0b"},
    "dark": {"series": "#3987e5", "accent": "#d95926", "surface": "#1a1a19",
             "grid": "#2c2c2a", "axis": "#383835", "muted": "#898781", "ink": "#ffffff"},
}

# ponytail: threshold hasil tuning PR-curve di model.ipynb (F1 terbaik).
# Default predict() memakai 0.5, yang menandai ~5% transaksi sah sebagai fraud.
# Tune ulang dan perbarui angka ini setiap model dilatih ulang; pindahkan ke
# dalam .pkl kalau ternyata sering berubah.
THRESHOLD = 0.9926
THRESHOLD_DIGITS = 4  # presisi tampilan slider; load_curve() menyamakan data ke sini

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

LANGUAGES = {"English": "en", "Bahasa Indonesia": "id"}

FIELDS = ("ex_type", "ex_amount", "ex_org_old", "ex_org_new", "ex_dst_old", "ex_dst_new")
LABELS = {v: k for k, v in TYPES.items()}  # PAYMENT -> "Payment"

# Bawaan form: satu baris fraud nyata yang memang tertangkap model, supaya
# tampilan pertama menunjukkan deteksi yang berhasil. Klik tombol setelahnya
# mengambil baris acak, termasuk yang gagal ditangkap.
DEFAULT_EXAMPLE = ("Transfer", 6_546_019.00, 6_546_019.00, 0.00, 0.00, 0.00)

# Diukur di test set yang sama, 1.899.033 baris. Aturan satu barisnya
# `oldbalanceOrg == amount`, memakai informasi pra-transaksi saja.
BASELINE_ROWS = [
    ("rule", 1.000, 0.977, 0.988),
    ("hgb", 0.937, 0.820, 0.875),
    ("this", 0.770, 0.539, 0.634),
]

# ponytail: angka di bawah di-hardcode dari model.ipynb. Dataset mentahnya
# 470 MB — app tidak boleh membacanya hanya untuk menampilkan statistik.
TYPE_ROWS = [
    ("CASH_OUT", 2_237_500, 4_116, 0.184),
    ("PAYMENT", 2_151_495, 0, 0.000),
    ("CASH_IN", 1_399_284, 0, 0.000),
    ("TRANSFER", 532_909, 4_097, 0.769),
    ("DEBIT", 41_432, 0, 0.000),
]

# (kolom, arti EN, arti ID, kunci peran)
COLUMN_ROWS = [
    ("step", "Hour of the simulation, 1 to 743 (31 days)", "Jam simulasi, 1 sampai 743 (31 hari)", "split"),
    ("type", "Transaction type, one of five categories", "Jenis transaksi, satu dari lima kategori", "feature"),
    ("amount", "Transaction amount", "Nominal transaksi", "feature"),
    ("nameOrig", "Sender account id", "ID akun pengirim", "dropped"),
    ("oldbalanceOrg", "Sender balance before the transaction", "Saldo pengirim sebelum transaksi", "feature"),
    ("newbalanceOrig", "Sender balance after the transaction", "Saldo pengirim setelah transaksi", "feature"),
    ("nameDest", "Recipient account id", "ID akun penerima", "dropped"),
    ("oldbalanceDest", "Recipient balance before the transaction", "Saldo penerima sebelum transaksi", "feature"),
    ("newbalanceDest", "Recipient balance after the transaction", "Saldo penerima setelah transaksi", "feature"),
    ("isFraud", "Whether the transaction was fraudulent", "Apakah transaksi itu fraud", "target"),
    ("isFlaggedFraud", "Flag raised by the simulator's own rule", "Tanda dari aturan bawaan simulator", "dropped"),
]

TEXT = {
    "en": {
        "title": "Fraud detection model",
        "subtitle": (
            "Logistic regression trained on 6.4 million simulated mobile-money transfers. "
            "Score a transaction, or read how the model and its data were built."
        ),
        "tabs": ["Predict", "The project", "The dataset"],
        "card": "Model card",
        "pr_auc_help": "The baseline for this dataset is 0.0024, so this model scores 280x higher.",
        "precision_help": "Of the transactions the model flags, 77% are fraud.",
        "recall_help": "The model catches 54% of real fraud and misses the rest.",
        "card_note": f"Logistic regression, decision threshold {THRESHOLD}.",
        "card_warning": "Trained on PaySim, a synthetic dataset. This is not a production fraud system.",
        "ex_caption": (
            "Each click loads a different real transaction from the held-out month, one the "
            "model never trained on. The dataset's own label is shown with the result."
        ),
        "btn_fraud": "Load a fraud example",
        "btn_legit": "Load a legitimate example",
        "fraud_caught": "The dataset labels this transaction fraud, and the model caught it.",
        "fraud_missed": (
            "The dataset labels this transaction fraud, and the model missed it. This is what a "
            "recall of 0.539 looks like from the inside: it happens to roughly every other "
            "fraudulent transaction."
        ),
        "legit_ok": "The dataset labels this transaction legitimate, and the model agrees.",
        "legit_false_alarm": (
            "The dataset labels this transaction legitimate, so the model raised a false alarm."
        ),
        "type_label": "Transaction type",
        "amount_label": "Transaction amount",
        "origin": "**Origin account**",
        "destination": "**Destination account**",
        "before": "Balance before",
        "after": "Balance after",
        "predict": "Predict",
        "result": "Result",
        "fraud_msg": "Predicted fraudulent. Fraud probability {p:.4f}, at or above the {t} threshold.",
        "legit_msg": "Predicted legitimate. Fraud probability {p:.4f}, below the {t} threshold.",
        "progress": "Fraud probability {p:.1%}",
        "no_fraud_note": (
            "The training data holds no fraud of type {kind}, so a fraud verdict here has "
            "nothing in the data behind it. The model still scores the balances, and unusual "
            "ones can carry it over the threshold anyway."
        ),
        "what_head": "What this is",
        "what_body": (
            "This classifier scores a single mobile-money transaction for fraud. The training "
            "data is public and synthetic, and no real payment system connects to this app."
        ),
        "how_head": "How the model works",
        "how_body": """
            One scikit-learn `Pipeline` holds every step, so each transformer fits on the
            training data alone and ships inside the same pickle file:

            1. `StandardScaler` scales the five numeric columns.
            2. `OneHotEncoder(drop="first")` encodes the transaction type.
            3. `LogisticRegression(class_weight="balanced")` reweights the rare class instead
               of resampling it.
        """,
        "decisions_head": "Design decisions",
        "split_head": "**Splitting by time**",
        "split_body": (
            "PaySim records each fraud as a pair of rows, a `TRANSFER` followed by a `CASH_OUT` "
            "sharing the same hour and amount, with identical sender balances in 98.4% of cases. "
            "A random split sent 1,665 of 4,080 pairs to opposite sides, so the model trained on "
            "near-copies of the rows it was later tested on. Cutting the data at hour 323 keeps "
            "each pair on one side and stops future transactions from reaching the training set."
        ),
        "metric_head": "**Scoring with PR-AUC**",
        "metric_body": (
            "At a 0.13% fraud rate, a model that labels everything legitimate scores 99.87% "
            f"accuracy. PR-AUC replaces accuracy here, and the decision threshold comes from the "
            f"precision-recall curve at {THRESHOLD} instead of the default 0.5. Tuning that "
            "threshold raised F1 from 0.04 to 0.635 and cut false positives from 101,499 to 734."
        ),
        "curve_head": "See the tradeoff",
        "curve_body": (
            "Every point on this curve is a threshold the model could use. Drag the slider to "
            "move the operating point and watch precision buy itself with recall."
        ),
        "curve_slider": "Decision threshold",
        "axis_recall": "Recall",
        "axis_precision": "Precision",
        "m_precision": "Precision",
        "m_recall": "Recall",
        "curve_caption": (
            "The shipped model sits at {t}, the threshold with the best F1. Anything to the "
            "right catches more fraud and troubles more honest customers."
        ),
        "learn_head": "Would more data help",
        "learn_body": (
            "Each point retrains the model on a random slice of the training window and scores "
            "it on the same held-out month. The slices all span the same period, so what moves "
            "is the amount of data, not how recent it is."
        ),
        "axis_rows": "Training transactions",
        "axis_prauc": "PR-AUC",
        "learn_caption": (
            "The line is flat. A hundredfold more data, from 45 thousand rows to 4.5 million, "
            "leaves PR-AUC around 0.7. Collecting more transactions is not the lever here, which "
            "is what sends the next section after a different model instead."
        ),
        "rule_head": "Checking against a trivial rule",
        "rule_body": (
            "Before trusting any of these numbers, it is worth asking whether the machine "
            "learning earns its place. PaySim injects fraud by draining an account completely, "
            "so a single equality catches it:"
        ),
        "rule_code": "flag = oldbalanceOrg == amount",
        "rule_result": (
            "Across the same held-out month, that line fires on 4,464 transactions. All 4,464 "
            "are fraud. It raises **not one false alarm** in 1,899,033 transactions, and none in "
            "the training period either, while catching 97.7% of all fraud. It uses only the "
            "balance and the amount, both known before the transaction runs."
        ),
        "rule_names": {"rule": "One-line rule", "hgb": "HistGradientBoosting",
                       "this": "Logistic regression (this app)"},
        "rule_headers": ["Approach", "Precision", "Recall", "F1"],
        "rule_why": (
            "The rule wins because the pattern is an equality between two features, and a linear "
            "model sums scaled features separately with no way to express *a equals b*. Adding "
            "the balance-error features changes nothing for the same reason: the condition is "
            "*equals zero*, not *larger is worse*."
        ),
        "rule_meaning": (
            "So what looks like signal is the simulator's own generating rule. Real fraudsters do "
            "not always take exactly the whole balance, and this rule would not survive contact "
            "with real transactions. The model here is worth building as an exercise; the "
            "benchmark it scores against is not a measure of how hard fraud detection is."
        ),
        "limits_head": "Where it falls short",
        "limits_body": f"""
            - **The model misses about 46% of fraud.** Recall sits at 0.539. Drop `THRESHOLD`
              below {THRESHOLD} to catch more fraud and flag more innocent customers.
            - **Logistic regression draws a straight line.** Gradient boosting reads this data
              better, at PR-AUC 0.955 against 0.674, though the rule above beats that too.
            - **A simulator produced the data.** Precision and recall on real transactions
              would differ.
            - **The model has no account history.** The features exclude sender and recipient
              ids, so it cannot tell that an account behaved oddly last week.
            - **It keeps extrapolating past what it saw.** Feed it a transfer of 10^15 and it
              returns 1.0000. The features are scaled linearly, so values far outside the
              training range push the score further instead of saturating. Fed uniformly random
              inputs, it calls 19% of them fraud, including types that carry no fraud at all.
        """,
        "data_head": "PaySim mobile-money transactions",
        "data_body": (
            "The PaySim simulator generates this dataset by modelling mobile-money traffic on an "
            "African payment network and injecting fraudulent agents into it. Banks cannot "
            "release real transaction data, so researchers train on this instead."
        ),
        "m_tx": "Transactions",
        "m_fraud": "Fraud cases",
        "m_rate": "Fraud rate",
        "m_span": "Time span",
        "m_span_value": "31 days",
        "imbalance": (
            "One fraud turns up in every 774 transactions. That imbalance shapes the split, the "
            "class weights, and the choice of metric."
        ),
        "types_head": "Transaction types",
        "types_body": (
            "Fraud appears in two of the five types. `TRANSFER` and `CASH_OUT` carry every "
            "fraudulent record, while `PAYMENT`, `CASH_IN` and `DEBIT` carry none. Fraudsters "
            "transfer a victim's balance out, then cash it."
        ),
        "cols_head": "Columns",
        "col_headers": ["Column", "Meaning", "Role"],
        "type_headers": ["Type", "Transactions", "Fraud", "Fraud rate (%)"],
        "roles": {"feature": "Feature", "dropped": "Dropped", "split": "Split only", "target": "Target"},
        "dropped_head": "Why three columns are dropped",
        "dropped_body": """
            - `nameOrig` and `nameDest` hold account ids. Senders are almost all unique, so the
              model would memorise identifiers instead of learning behaviour.
            - `isFlaggedFraud` holds the simulator's own fraud flag. Feeding a fraud label into
              a fraud model leaks the answer, and it fires on 16 of 6.36 million rows.
            - `step` stays in the table and never reaches the model. It marks the train/test
              boundary, and the `ColumnTransformer` leaves it out of the features.
        """,
        "median_note": (
            "Fraudulent transactions run much larger: a median of 441,424 against 74,872 across "
            "all transactions."
        ),
    },
    "id": {
        "title": "Model deteksi fraud",
        "subtitle": (
            "Regresi logistik yang dilatih pada 6,4 juta transaksi uang elektronik simulasi. "
            "Nilai sebuah transaksi, atau baca cara model dan datanya dibangun."
        ),
        "tabs": ["Prediksi", "Tentang project", "Tentang dataset"],
        "card": "Kartu model",
        "pr_auc_help": "Baseline dataset ini 0,0024, jadi model ini unggul 280 kali lipat.",
        "precision_help": "Dari transaksi yang ditandai model, 77% benar-benar fraud.",
        "recall_help": "Model menangkap 54% fraud asli dan melewatkan sisanya.",
        "card_note": f"Regresi logistik, ambang keputusan {THRESHOLD}.",
        "card_warning": "Dilatih pada PaySim, dataset sintetis. Ini bukan sistem fraud produksi.",
        "ex_caption": (
            "Tiap klik memuat transaksi nyata yang berbeda dari bulan uji, yang tak pernah "
            "dilatihkan ke model. Label asli dari dataset ditampilkan bersama hasilnya."
        ),
        "btn_fraud": "Muat contoh fraud",
        "btn_legit": "Muat contoh sah",
        "fraud_caught": "Dataset melabeli transaksi ini fraud, dan model menangkapnya.",
        "fraud_missed": (
            "Dataset melabeli transaksi ini fraud, dan model melewatkannya. Beginilah wujud "
            "recall 0,539 dari dekat: kira-kira terjadi pada satu dari setiap dua transaksi "
            "fraud."
        ),
        "legit_ok": "Dataset melabeli transaksi ini sah, dan model sependapat.",
        "legit_false_alarm": (
            "Dataset melabeli transaksi ini sah, jadi model salah menuduh."
        ),
        "type_label": "Jenis transaksi",
        "amount_label": "Nominal transaksi",
        "origin": "**Akun pengirim**",
        "destination": "**Akun penerima**",
        "before": "Saldo sebelum",
        "after": "Saldo sesudah",
        "predict": "Prediksi",
        "result": "Hasil",
        "fraud_msg": "Diprediksi fraud. Probabilitas fraud {p:.4f}, sama atau di atas ambang {t}.",
        "legit_msg": "Diprediksi sah. Probabilitas fraud {p:.4f}, di bawah ambang {t}.",
        "progress": "Probabilitas fraud {p:.1%}",
        "no_fraud_note": (
            "Data latih tidak memuat satu pun fraud berjenis {kind}, jadi vonis fraud di sini "
            "tidak berdasar apa pun di data. Model tetap menilai saldonya, dan angka yang tidak "
            "lazim tetap bisa mendorongnya melewati ambang."
        ),
        "what_head": "Ini apa",
        "what_body": (
            "Pengklasifikasi ini menilai satu transaksi uang elektronik untuk mendeteksi fraud. "
            "Data latihnya publik dan sintetis, dan tidak ada sistem pembayaran nyata yang "
            "terhubung ke app ini."
        ),
        "how_head": "Cara kerja model",
        "how_body": """
            Satu `Pipeline` scikit-learn memuat seluruh langkah, sehingga tiap transformer
            dilatih hanya pada data latih dan ikut tersimpan dalam file pickle yang sama:

            1. `StandardScaler` menskalakan lima kolom numerik.
            2. `OneHotEncoder(drop="first")` mengubah jenis transaksi jadi kolom biner.
            3. `LogisticRegression(class_weight="balanced")` memberi bobot lebih pada kelas
               langka alih-alih menduplikasi barisnya.
        """,
        "decisions_head": "Keputusan desain",
        "split_head": "**Memecah data berdasarkan waktu**",
        "split_body": (
            "PaySim mencatat tiap fraud sebagai sepasang baris, satu `TRANSFER` lalu satu "
            "`CASH_OUT` dengan jam dan nominal sama, dan saldo pengirim identik pada 98,4% kasus. "
            "Pemecahan acak melempar 1.665 dari 4.080 pasangan ke sisi berlawanan, jadi model "
            "berlatih pada kembaran baris yang kemudian dipakai mengujinya. Memotong data di jam "
            "323 menjaga tiap pasangan tetap satu sisi dan menahan transaksi masa depan masuk ke "
            "data latih."
        ),
        "metric_head": "**Menilai dengan PR-AUC**",
        "metric_body": (
            "Pada tingkat fraud 0,13%, model yang menilai semuanya sah tetap meraih akurasi "
            f"99,87%. PR-AUC menggantikan akurasi di sini, dan ambang keputusan diambil dari "
            f"kurva precision-recall di {THRESHOLD}, bukan 0,5 bawaan. Tuning ambang itu "
            "menaikkan F1 dari 0,04 ke 0,635 dan memangkas false positive dari 101.499 jadi 734."
        ),
        "curve_head": "Lihat trade-off-nya",
        "curve_body": (
            "Tiap titik di kurva ini satu threshold yang bisa dipakai model. Geser slider untuk "
            "memindahkan titik operasi dan lihat precision membeli dirinya dengan recall."
        ),
        "curve_slider": "Ambang keputusan",
        "axis_recall": "Recall",
        "axis_precision": "Precision",
        "m_precision": "Precision",
        "m_recall": "Recall",
        "curve_caption": (
            "Model yang dipakai duduk di {t}, threshold dengan F1 terbaik. Bergeser ke kanan "
            "menangkap lebih banyak fraud dan mengganggu lebih banyak nasabah jujur."
        ),
        "learn_head": "Apakah menambah data menolong",
        "learn_body": (
            "Tiap titik melatih ulang model pada potongan acak jendela latih lalu menilainya di "
            "bulan uji yang sama. Semua potongan menutupi rentang waktu yang sama, jadi yang "
            "berubah cuma banyaknya data, bukan seberapa barunya."
        ),
        "axis_rows": "Transaksi latih",
        "axis_prauc": "PR-AUC",
        "learn_caption": (
            "Garisnya datar. Data seratus kali lipat, dari 45 ribu baris ke 4,5 juta, "
            "meninggalkan PR-AUC di sekitar 0,7. Mengumpulkan lebih banyak transaksi bukan "
            "tuasnya di sini, dan itu yang mengarahkan bagian berikutnya ke model lain."
        ),
        "rule_head": "Menguji lawan aturan sepele",
        "rule_body": (
            "Sebelum mempercayai angka-angka di atas, layak ditanya apakah machine learning-nya "
            "memang perlu. PaySim menyisipkan fraud dengan menguras akun sampai habis, jadi satu "
            "kesetaraan sudah menangkapnya:"
        ),
        "rule_code": "flag = oldbalanceOrg == amount",
        "rule_result": (
            "Di bulan uji yang sama, baris itu menyala pada 4.464 transaksi. Keempat ribu empat "
            "ratus enam puluh empat-empatnya fraud. **Nol salah tuduh** dari 1.899.033 transaksi, "
            "nol juga di periode latih, sambil menangkap 97,7% seluruh fraud. Yang dipakai cuma "
            "saldo dan nominal, dua-duanya sudah diketahui sebelum transaksi berjalan."
        ),
        "rule_names": {"rule": "Aturan satu baris", "hgb": "HistGradientBoosting",
                       "this": "Regresi logistik (app ini)"},
        "rule_headers": ["Pendekatan", "Precision", "Recall", "F1"],
        "rule_why": (
            "Aturan itu menang karena polanya adalah kesetaraan antar dua fitur, sementara model "
            "linear menjumlahkan fitur terskala secara terpisah tanpa cara mengungkapkan *a sama "
            "dengan b*. Menambah fitur error-saldo pun tidak mengubah apa pun, dengan alasan yang "
            "sama: syaratnya *sama dengan nol*, bukan *makin besar makin fraud*."
        ),
        "rule_meaning": (
            "Jadi yang tampak seperti sinyal sebenarnya aturan pembangkit simulatornya. Penipu "
            "sungguhan tidak selalu mengambil tepat seluruh saldo, dan aturan ini tidak akan "
            "bertahan di data nyata. Model di sini layak dibangun sebagai latihan; tolok ukur "
            "yang dipakainya bukan ukuran seberapa sulit deteksi fraud sebenarnya."
        ),
        "limits_head": "Batas kemampuannya",
        "limits_body": f"""
            - **Model melewatkan sekitar 46% fraud.** Recall-nya 0,539. Turunkan `THRESHOLD`
              di bawah {THRESHOLD} untuk menangkap lebih banyak fraud sekaligus menandai lebih
              banyak nasabah yang tidak bersalah.
            - **Regresi logistik menarik garis lurus.** Gradient boosting membaca data ini lebih
              baik, PR-AUC 0,955 berbanding 0,674, walau aturan di atas tetap mengalahkannya.
            - **Datanya buatan simulator.** Precision dan recall pada transaksi nyata akan
              berbeda.
            - **Model tidak punya riwayat akun.** Fiturnya tidak memuat ID pengirim dan
              penerima, jadi model tidak tahu sebuah akun berperilaku aneh minggu lalu.
            - **Model terus mengekstrapolasi di luar data yang dilihatnya.** Beri transfer
              sebesar 10^15 dan hasilnya 1.0000. Fiturnya diskalakan linear, jadi nilai jauh di
              luar rentang latih mendorong skor makin tinggi alih-alih jenuh. Diberi input acak
              seragam, 19% di antaranya divonis fraud, termasuk jenis yang sama sekali tidak
              memuat fraud.
        """,
        "data_head": "Transaksi uang elektronik PaySim",
        "data_body": (
            "Simulator PaySim menghasilkan dataset ini dengan memodelkan lalu lintas uang "
            "elektronik di sebuah jaringan pembayaran Afrika lalu menyisipkan pelaku fraud ke "
            "dalamnya. Bank tidak boleh merilis data transaksi asli, jadi peneliti memakai ini."
        ),
        "m_tx": "Transaksi",
        "m_fraud": "Kasus fraud",
        "m_rate": "Tingkat fraud",
        "m_span": "Rentang waktu",
        "m_span_value": "31 hari",
        "imbalance": (
            "Satu fraud muncul tiap 774 transaksi. Ketimpangan itu menentukan cara pemecahan "
            "data, bobot kelas, dan pilihan metrik."
        ),
        "types_head": "Jenis transaksi",
        "types_body": (
            "Fraud muncul di dua dari lima jenis. `TRANSFER` dan `CASH_OUT` memuat seluruh "
            "catatan fraud, sementara `PAYMENT`, `CASH_IN` dan `DEBIT` tidak memuat satu pun. "
            "Pelaku memindahkan saldo korban keluar, lalu menariknya jadi tunai."
        ),
        "cols_head": "Kolom",
        "col_headers": ["Kolom", "Arti", "Peran"],
        "type_headers": ["Jenis", "Transaksi", "Fraud", "Tingkat fraud (%)"],
        "roles": {"feature": "Fitur", "dropped": "Dibuang", "split": "Hanya split", "target": "Target"},
        "dropped_head": "Alasan tiga kolom dibuang",
        "dropped_body": """
            - `nameOrig` dan `nameDest` berisi ID akun. Pengirim hampir seluruhnya unik, jadi
              model akan menghafal identitas alih-alih mempelajari perilaku.
            - `isFlaggedFraud` berisi tanda fraud bawaan simulator. Memberi label fraud ke model
              fraud membocorkan jawabannya, dan tanda itu menyala pada 16 dari 6,36 juta baris.
            - `step` tetap ada di tabel dan tidak pernah sampai ke model. Kolom itu menandai
              batas latih/uji, dan `ColumnTransformer` mengeluarkannya dari fitur.
        """,
        "median_note": (
            "Transaksi fraud juga jauh lebih besar: median 441.424 berbanding 74.872 pada seluruh "
            "transaksi."
        ),
    },
}


@st.cache_resource
def load_model():
    # Pickle di-load dari artefak yang dihasilkan model.ipynb di repo ini dan
    # ikut ter-commit, bukan unggahan user. Jangan arahkan MODEL_PATH ke file
    # dari sumber luar: unpickle menjalankan kode sembarang.
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_examples():
    # Semua baris berasal dari periode uji, jadi tak satu pun pernah dilatih.
    return pd.read_csv(EXAMPLES_PATH)


@st.cache_data
def load_learning():
    return pd.read_csv(LEARNING_PATH)


@st.cache_data
def load_curve():
    # select_slider menyimpan state sebagai hasil format_func lalu memetakannya
    # balik lewat label, jadi label yang bentrok menunjuk titik yang salah.
    # Bulatkan ke presisi yang sama dengan tampilannya, baru buang duplikat:
    # label dijamin unik, bukan sekadar berharap unik.
    curve = pd.read_csv(CURVE_PATH)
    curve["threshold"] = curve["threshold"].round(THRESHOLD_DIGITS)
    return curve.drop_duplicates("threshold").sort_values("threshold").reset_index(drop=True)


model = load_model()

trained = set(model.named_steps["preprocessor"].named_transformers_["cat"].categories_[0])
assert set(TYPES.values()) <= trained, f"kategori tak dikenal model: {set(TYPES.values()) - trained}"

with st.sidebar:
    choice = st.segmented_control("Language / Bahasa", list(LANGUAGES), default="English")
    lang = LANGUAGES.get(choice, "en")
    t = TEXT[lang]

    st.subheader(t["card"])
    st.metric("PR-AUC", "0.674", help=t["pr_auc_help"])
    st.metric("Precision", "0.771", help=t["precision_help"])
    st.metric("Recall", "0.539", help=t["recall_help"])
    st.caption(t["card_note"])
    st.caption(t["card_warning"])

st.title(t["title"])
st.caption(t["subtitle"])

predict_tab, project_tab, data_tab = st.tabs(t["tabs"])

with predict_tab:
    examples = load_examples()

    for field, value in zip(FIELDS, DEFAULT_EXAMPLE):
        st.session_state.setdefault(field, value)
    st.session_state.setdefault("ex_actual", (1, DEFAULT_EXAMPLE))

    def use_example(is_fraud):
        row = examples[examples.isFraud == is_fraud].sample(1).iloc[0]
        values = (LABELS[row.type], float(row.amount), float(row.oldbalanceOrg),
                  float(row.newbalanceOrig), float(row.oldbalanceDest), float(row.newbalanceDest))
        for field, value in zip(FIELDS, values):
            st.session_state[field] = value
        # Label asli disimpan BESERTA nilainya, supaya keterangannya hilang
        # sendiri begitu user mengubah salah satu angka.
        st.session_state["ex_actual"] = (is_fraud, values)

    st.caption(t["ex_caption"])
    with st.container(horizontal=True):
        st.button(t["btn_fraud"], icon=":material/gpp_bad:", on_click=use_example, args=(1,))
        st.button(t["btn_legit"], icon=":material/verified_user:", on_click=use_example, args=(0,))

    with st.form("transaction"):
        transaction_type = st.selectbox(t["type_label"], list(TYPES), key="ex_type")
        amount = st.number_input(t["amount_label"], min_value=0.0, step=0.01, key="ex_amount")

        left, right = st.columns(2)
        with left:
            st.markdown(t["origin"])
            oldbalanceorg = st.number_input(t["before"], min_value=0.0, step=0.01, key="ex_org_old")
            newbalanceorg = st.number_input(t["after"], min_value=0.0, step=0.01, key="ex_org_new")
        with right:
            st.markdown(t["destination"])
            oldbalancedest = st.number_input(
                t["before"], min_value=0.0, step=0.01, key="ex_dst_old"
            )
            newbalancedest = st.number_input(
                t["after"], min_value=0.0, step=0.01, key="ex_dst_new"
            )

        submitted = st.form_submit_button(
            t["predict"], icon=":material/play_arrow:", type="primary"
        )

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

        st.subheader(t["result"])
        if fraud_proba >= THRESHOLD:
            st.error(
                t["fraud_msg"].format(p=fraud_proba, t=THRESHOLD), icon=":material/gpp_bad:"
            )
        else:
            st.success(
                t["legit_msg"].format(p=fraud_proba, t=THRESHOLD), icon=":material/verified_user:"
            )

        # Ditampilkan hanya kalau form masih persis seperti saat contoh dimuat.
        loaded = st.session_state.get("ex_actual")
        current = (transaction_type, amount, oldbalanceorg, newbalanceorg,
                   oldbalancedest, newbalancedest)
        if loaded and loaded[1] == current:
            flagged = fraud_proba >= THRESHOLD
            if loaded[0] == 1:
                verdict = "fraud_caught" if flagged else "fraud_missed"
            else:
                verdict = "legit_false_alarm" if flagged else "legit_ok"
            st.info(t[verdict], icon=":material/fact_check:")

        st.progress(fraud_proba, text=t["progress"].format(p=fraud_proba))

        if TYPES[transaction_type] not in ("TRANSFER", "CASH_OUT"):
            st.caption(t["no_fraud_note"].format(kind=TYPES[transaction_type]))

with project_tab:
    st.subheader(t["what_head"])
    st.markdown(t["what_body"])

    st.subheader(t["how_head"])
    st.markdown(t["how_body"])

    st.subheader(t["decisions_head"])
    with st.container(border=True):
        st.markdown(t["split_head"])
        st.markdown(t["split_body"])
    with st.container(border=True):
        st.markdown(t["metric_head"])
        st.markdown(t["metric_body"])

    st.subheader(t["curve_head"])
    st.markdown(t["curve_body"])

    curve = load_curve()
    options = curve["threshold"].tolist()
    picked = st.select_slider(
        t["curve_slider"],
        options=options,
        value=min(options, key=lambda v: abs(v - THRESHOLD)),
        format_func=lambda v: f"{v:.{THRESHOLD_DIGITS}f}",
    )
    row = curve.loc[curve["threshold"] == picked].iloc[0]
    f1 = 2 * row.precision * row.recall / (row.precision + row.recall + 1e-12)

    scores = st.columns(3)
    scores[0].metric(t["m_precision"], f"{row.precision:.3f}")
    scores[1].metric(t["m_recall"], f"{row.recall:.3f}")
    scores[2].metric("F1", f"{f1:.3f}")

    p = PALETTE["dark" if st.context.theme.get("type") == "dark" else "light"]
    axis = alt.Axis(
        format=".0%", labelColor=p["muted"], titleColor=p["muted"],
        tickColor=p["axis"], domainColor=p["axis"], gridColor=p["grid"],
    )
    span = alt.Scale(domain=[0, 1], nice=False)

    line = alt.Chart(curve).mark_line(strokeWidth=2, color=p["series"]).encode(
        x=alt.X("recall:Q", title=t["axis_recall"], axis=axis, scale=span),
        y=alt.Y("precision:Q", title=t["axis_precision"], axis=axis, scale=span),
    )
    # Lapisan hover transparan: kurva jadi bisa ditanya tanpa menambah mark terlihat.
    hover = alt.Chart(curve).mark_circle(size=90, opacity=0).encode(
        x=alt.X("recall:Q", scale=span),
        y=alt.Y("precision:Q", scale=span),
        tooltip=[
            alt.Tooltip("threshold:Q", title=t["curve_slider"], format=f".{THRESHOLD_DIGITS}f"),
            alt.Tooltip("precision:Q", title=t["m_precision"], format=".3f"),
            alt.Tooltip("recall:Q", title=t["m_recall"], format=".3f"),
        ],
    )
    here = pd.DataFrame([{"recall": row.recall, "precision": row.precision,
                          "label": f"{picked:.{THRESHOLD_DIGITS}f}"}])
    dot = alt.Chart(here).mark_point(
        size=170, filled=True, color=p["accent"], stroke=p["surface"], strokeWidth=2,
    ).encode(x=alt.X("recall:Q", scale=span), y=alt.Y("precision:Q", scale=span))
    # Label langsung: titik operasi tidak boleh dikenali dari warna saja.
    tag = alt.Chart(here).mark_text(
        dx=10, dy=-12, align="left", fontSize=12, color=p["ink"],
    ).encode(x=alt.X("recall:Q", scale=span), y=alt.Y("precision:Q", scale=span), text="label:N")

    st.altair_chart(
        (line + hover + dot + tag).properties(height=320, background=p["surface"]),
        theme=None,
    )
    st.caption(t["curve_caption"].format(t=THRESHOLD))

    st.subheader(t["learn_head"])
    st.markdown(t["learn_body"])

    learn = load_learning()
    rows_axis = alt.Axis(labelColor=p["muted"], titleColor=p["muted"], tickColor=p["axis"],
                         domainColor=p["axis"], gridColor=p["grid"], format="~s")
    pct_axis = alt.Axis(format=".0%", labelColor=p["muted"], titleColor=p["muted"],
                        tickColor=p["axis"], domainColor=p["axis"], gridColor=p["grid"])
    # Sumbu y penuh 0-1: memotongnya ke 0.6-0.75 akan membesar-besarkan derau
    # jadi seolah tren, padahal justru kedatarannya yang jadi temuan.
    enc = dict(
        x=alt.X("rows:Q", title=t["axis_rows"], axis=rows_axis,
                scale=alt.Scale(type="log", nice=False)),
        y=alt.Y("test_pr_auc:Q", title=t["axis_prauc"], axis=pct_axis,
                scale=alt.Scale(domain=[0, 1], nice=False)),
    )
    learn_line = alt.Chart(learn).mark_line(strokeWidth=2, color=p["series"]).encode(**enc)
    learn_dots = alt.Chart(learn).mark_point(
        size=90, filled=True, color=p["series"], stroke=p["surface"], strokeWidth=2,
    ).encode(
        **enc,
        tooltip=[alt.Tooltip("rows:Q", title=t["axis_rows"], format=","),
                 alt.Tooltip("fraud:Q", title=t["m_fraud"], format=","),
                 alt.Tooltip("test_pr_auc:Q", title="PR-AUC", format=".3f")],
    )
    st.altair_chart(
        (learn_line + learn_dots).properties(height=280, background=p["surface"]), theme=None
    )
    st.caption(t["learn_caption"])

    st.subheader(t["rule_head"])
    st.markdown(t["rule_body"])
    st.code(t["rule_code"], language="python")
    st.markdown(t["rule_result"])
    st.dataframe(
        pd.DataFrame([(t["rule_names"][k], f"{p:.3f}", f"{r:.3f}", f"{f:.3f}")
                      for k, p, r, f in BASELINE_ROWS], columns=t["rule_headers"]),
        hide_index=True, width="stretch",
    )
    st.markdown(t["rule_why"])
    st.markdown(t["rule_meaning"])

    st.subheader(t["limits_head"])
    st.markdown(t["limits_body"])

with data_tab:
    st.subheader(t["data_head"])
    st.markdown(t["data_body"])

    cols = st.columns(4)
    cols[0].metric(t["m_tx"], "6.36M")
    cols[1].metric(t["m_fraud"], "8,213")
    cols[2].metric(t["m_rate"], "0.13%")
    cols[3].metric(t["m_span"], t["m_span_value"])

    st.markdown(t["imbalance"])

    st.subheader(t["types_head"])
    st.dataframe(
        pd.DataFrame(TYPE_ROWS, columns=t["type_headers"]), hide_index=True, width="stretch"
    )
    st.markdown(t["types_body"])

    st.subheader(t["cols_head"])
    meaning = 1 if lang == "en" else 2
    st.dataframe(
        pd.DataFrame(
            [(row[0], row[meaning], t["roles"][row[3]]) for row in COLUMN_ROWS],
            columns=t["col_headers"],
        ),
        hide_index=True,
        width="stretch",
    )

    with st.expander(t["dropped_head"], icon=":material/delete:"):
        st.markdown(t["dropped_body"])

    st.caption(t["median_note"])
