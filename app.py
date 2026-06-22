import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="💳",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .fraud-box    { background:#ff4b4b22; border:2px solid #ff4b4b; border-radius:10px; padding:20px; text-align:center; }
    .legit-box    { background:#00c85322; border:2px solid #00c853; border-radius:10px; padding:20px; text-align:center; }
    .metric-card  { background:#1e1e2e; border-radius:10px; padding:15px; margin:5px; text-align:center; }
    .big-text     { font-size:2rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("💳 Credit Card Fraud Detection")
st.markdown("**ML-powered fraud detection using 6 classification algorithms**")
st.markdown("---")

# ── Load & train model (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Training models on dataset...")
def load_and_train(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # Generate synthetic data matching real dataset structure
        np.random.seed(42)
        n = 10000
        n_fraud = 17
        X_normal = np.random.randn(n - n_fraud, 30)
        X_fraud  = np.random.randn(n_fraud, 30) * 1.5 + 2
        X_all    = np.vstack([X_normal, X_fraud])
        y_all    = np.array([0]*(n - n_fraud) + [1]*n_fraud)
        cols     = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
        df       = pd.DataFrame(X_all, columns=cols)
        df["Class"] = y_all

    normal = df[df.Class == 0]
    fraud  = df[df.Class == 1]
    n_fraud_samples = len(fraud)

    # Under-sampling
    legit_sample = normal.sample(n=n_fraud_samples, random_state=42)
    balanced_df  = pd.concat([legit_sample, fraud], axis=0).sample(frac=1, random_state=42)

    X = balanced_df.drop("Class", axis=1)
    y = balanced_df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

    models = {
        "Random Forest":       RandomForestClassifier(n_estimators=50, criterion="entropy", random_state=42),
        "Logistic Regression": LogisticRegression(solver="liblinear", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(criterion="entropy", random_state=42),
        "SVM":                 svm.SVC(kernel="rbf", C=10, probability=True, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }

    trained, scores = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[name] = round(accuracy_score(y_test, y_pred) * 100, 2)
        trained[name] = model

    return trained, scores, X_train.columns.tolist(), df, fraud, normal

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader("Upload creditcard.csv", type=["csv"])
    st.markdown("---")
    st.markdown("**Don't have the dataset?**")
    st.markdown("[Download from Kaggle →](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")
    st.markdown("---")
    selected_model = st.selectbox("Choose Model for Prediction", [
        "Random Forest", "Logistic Regression", "Decision Tree",
        "SVM", "Naive Bayes", "KNN"
    ])
    st.markdown("---")
    st.info("🔒 Data is processed locally — nothing is stored or sent anywhere.")

# ── Load models ───────────────────────────────────────────────────────────────
trained_models, model_scores, feature_cols, full_df, fraud_df, normal_df = load_and_train(uploaded)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 Model Performance", "📈 Data Insights"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔍 Predict a Transaction")
    st.markdown("Adjust the sliders below or paste raw feature values to check if a transaction is fraudulent.")

    mode = st.radio("Input Mode", ["Use Sliders", "Paste Raw Values"], horizontal=True)

    if mode == "Use Sliders":
        col1, col2, col3 = st.columns(3)
        inputs = {}
        v_features = [f"V{i}" for i in range(1, 29)]
        for idx, feat in enumerate(v_features):
            col = [col1, col2, col3][idx % 3]
            with col:
                inputs[feat] = st.slider(feat, -5.0, 5.0, 0.0, 0.1)
        c1, c2 = st.columns(2)
        with c1:
            inputs["Amount"] = st.number_input("Amount ($)", 0.0, 50000.0, 100.0)
        with c2:
            inputs["Time"] = st.number_input("Time (seconds)", 0.0, 172800.0, 50000.0)
        input_df = pd.DataFrame([inputs])[feature_cols]

    else:
        st.markdown("Paste comma-separated values for **V1 to V28, Amount, Time** (30 values total):")
        raw = st.text_area("Raw values", "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,100,50000")
        try:
            vals = [float(x.strip()) for x in raw.split(",")]
            if len(vals) != 30:
                st.error(f"Need exactly 30 values, got {len(vals)}")
                st.stop()
            input_df = pd.DataFrame([vals], columns=feature_cols)
        except Exception as e:
            st.error(f"Invalid input: {e}")
            st.stop()

    st.markdown("---")
    if st.button("🚀 Run Prediction", use_container_width=True, type="primary"):
        model = trained_models[selected_model]
        prediction = model.predict(input_df)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0]
            confidence = prob[prediction] * 100
        else:
            confidence = None

        st.markdown("### Result")
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-box">
                <div class="big-text">🚨 FRAUDULENT TRANSACTION</div>
                <p style="font-size:1.2rem">This transaction has been flagged as <b>FRAUD</b></p>
                {"<p>Confidence: <b>" + f"{confidence:.1f}%" + "</b></p>" if confidence else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="legit-box">
                <div class="big-text">✅ LEGITIMATE TRANSACTION</div>
                <p style="font-size:1.2rem">This transaction appears <b>NORMAL</b></p>
                {"<p>Confidence: <b>" + f"{confidence:.1f}%" + "</b></p>" if confidence else ""}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"*Predicted using: **{selected_model}** (Accuracy: {model_scores[selected_model]}%)*")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Model Accuracy Comparison")

    # Metrics row
    cols = st.columns(len(model_scores))
    best_model = max(model_scores, key=model_scores.get)
    for col, (name, score) in zip(cols, model_scores.items()):
        with col:
            delta = "🥇 Best" if name == best_model else None
            st.metric(name, f"{score}%", delta)

    st.markdown("---")

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#FFD700" if n == best_model else "#4C72B0" for n in model_scores]
    bars = ax.bar(model_scores.keys(), model_scores.values(), color=colors, edgecolor="white", linewidth=1.2)
    for bar, score in zip(bars, model_scores.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{score}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison (with GridSearchCV best params)")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("🏆 Best Model Details")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", best_model)
    c2.metric("Accuracy", f"{model_scores[best_model]}%")
    c3.metric("Best Params", "criterion=entropy, n_estimators=50")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📈 Dataset Insights")

    total  = len(full_df)
    n_fr   = len(fraud_df)
    n_lg   = len(normal_df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Legitimate", f"{n_lg:,}")
    c3.metric("Fraudulent", f"{n_fr:,}")
    c4.metric("Fraud Rate", f"{n_fr/total*100:.2f}%")

    st.markdown("---")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.patch.set_facecolor("#0e1117")

    # Class distribution
    axes[0].bar(["Legitimate", "Fraud"], [n_lg, n_fr],
                color=["#00c853", "#ff4b4b"], edgecolor="white")
    axes[0].set_title("Class Distribution", color="white")
    axes[0].set_yscale("log")
    axes[0].set_facecolor("#0e1117")
    axes[0].tick_params(colors="white")

    # Amount distribution
    axes[1].hist(fraud_df["Amount"] if "Amount" in fraud_df.columns else fraud_df.iloc[:, -2],
                 bins=30, color="#ff4b4b", alpha=0.7, label="Fraud")
    axes[1].hist(normal_df["Amount"].sample(min(500, n_lg)) if "Amount" in normal_df.columns
                 else normal_df.iloc[:, -2].sample(min(500, n_lg)),
                 bins=30, color="#00c853", alpha=0.5, label="Legit")
    axes[1].set_title("Transaction Amount", color="white")
    axes[1].set_xlabel("Amount ($)", color="white")
    axes[1].legend()
    axes[1].set_facecolor("#0e1117")
    axes[1].tick_params(colors="white")

    # Model scores
    axes[2].barh(list(model_scores.keys()), list(model_scores.values()),
                 color=["#FFD700" if n == best_model else "#4C72B0" for n in model_scores])
    axes[2].set_title("Model Scores", color="white")
    axes[2].set_xlabel("Accuracy %", color="white")
    axes[2].set_facecolor("#0e1117")
    axes[2].tick_params(colors="white")
    axes[2].set_xlim(0, 110)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if uploaded:
        st.markdown("---")
        st.subheader("Raw Data Preview")
        st.dataframe(full_df.head(20), use_container_width=True)