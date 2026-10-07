"""
Titanic Survival Prediction — Interactive ML Comparison App
Mini Project — Machine Learning

Compares Decision Tree, Random Forest, Naive Bayes, and KNN
to predict whether a Titanic passenger survived.
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="⚓", layout="wide")

# --------------------------------------------------------------------------
# THEME — dark nautical palette, distinct from a typical light dashboard
# --------------------------------------------------------------------------
THEME_CSS = """
<style>
.stApp {
    background: linear-gradient(180deg, #0b1e33 0%, #0f2942 100%);
}
[data-testid="stSidebar"] {
    background: #081726;
    border-right: 1px solid #1c3a57;
}
h1, h2, h3, h4, p, span, label, .stMarkdown {
    color: #e8f1f8 !important;
}
.hero {
    background: linear-gradient(120deg, #0d3b66 0%, #14507a 60%, #d4a017 160%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 22px;
    border: 1px solid #2a5a80;
}
.hero h1 {
    margin: 0;
    font-size: 2.1rem;
}
.hero p {
    margin-top: 8px;
    color: #cfe3f0 !important;
    font-size: 0.98rem;
}
.metric-card {
    background: #102b45;
    border: 1px solid #21496b;
    border-radius: 12px;
    padding: 16px 18px;
    text-align: center;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #f2c14e !important;
}
.metric-card .label {
    font-size: 0.85rem;
    color: #a9c4d8 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.section-badge {
    display: inline-block;
    background: #d4a017;
    color: #0b1e33 !important;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-bottom: 6px;
}
.stButton>button {
    background: #d4a017;
    color: #0b1e33;
    font-weight: 700;
    border-radius: 8px;
    border: none;
}
.stButton>button:hover {
    background: #f2c14e;
    color: #0b1e33;
}
div[data-testid="stExpander"] {
    background: #0e2438;
    border: 1px solid #21496b;
    border-radius: 12px;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"
ACCENT_SEQUENCE = ["#d4a017", "#4fb0c6", "#e07a5f", "#81b29a"]

ALGORITHMS = {
    "Decision Tree": {
        "needs_scaling": False,
        "build": lambda p: DecisionTreeClassifier(max_depth=p["dt_depth"], random_state=42),
    },
    "Random Forest": {
        "needs_scaling": False,
        "build": lambda p: RandomForestClassifier(n_estimators=p["rf_trees"], random_state=42),
    },
    "Naive Bayes": {
        "needs_scaling": False,
        "build": lambda p: GaussianNB(),
    },
    "KNN": {
        "needs_scaling": True,
        "build": lambda p: KNeighborsClassifier(n_neighbors=p["knn_k"]),
    },
}

ID_LIKE_COLS = ["passengerid", "name", "ticket", "cabin", "id"]


@st.cache_data
def load_default_dataset():
    return pd.read_csv("titanic_dataset.csv")


def preprocess(df, target_col):
    df = df.copy().dropna(subset=[target_col])
    y = df[target_col]
    if y.dtype == object:
        y = LabelEncoder().fit_transform(y)
    X = df.drop(columns=[target_col])

    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown")
        else:
            X[col] = X[col].fillna(X[col].median())

    encoders = {}
    for col in X.columns:
        if X[col].dtype == object:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    return X, y, encoders


def metric_card(col, label, value):
    col.markdown(
        f"""<div class="metric-card"><div class="value">{value}</div>
        <div class="label">{label}</div></div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# SIDEBAR — laid out as numbered "voyage stages" instead of plain widgets
# --------------------------------------------------------------------------
st.sidebar.markdown("## ⚓ Control Deck")

st.sidebar.markdown("**Stage 1 · Load Data**")
uploaded_file = st.sidebar.file_uploader("Upload a CSV", type=["csv"], label_visibility="collapsed")
use_sample = st.sidebar.checkbox("Use built-in sample dataset", value=uploaded_file is None)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
    data_source = f"Uploaded file: {uploaded_file.name}"
elif use_sample:
    raw_df = load_default_dataset()
    data_source = "Built-in synthetic Titanic dataset (900 passengers)"
else:
    st.sidebar.warning("Upload a CSV or enable the sample dataset to continue.")
    st.stop()

drop_cols = [c for c in raw_df.columns if c.lower() in ID_LIKE_COLS]
raw_df = raw_df.drop(columns=drop_cols)

st.sidebar.markdown("**Stage 2 · Target Column**")
default_target = "Survived" if "Survived" in raw_df.columns else raw_df.columns[-1]
target_col = st.sidebar.selectbox(
    "Column to predict", options=list(raw_df.columns),
    index=list(raw_df.columns).index(default_target), label_visibility="collapsed",
)

st.sidebar.markdown("**Stage 3 · Algorithms**")
selected_algos = st.sidebar.multiselect(
    "Pick algorithms", options=list(ALGORITHMS.keys()),
    default=["Decision Tree", "Random Forest", "Naive Bayes", "KNN"],
    label_visibility="collapsed",
)

st.sidebar.markdown("**Stage 4 · Split & Tuning**")
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20, step=5) / 100
with st.sidebar.expander("Fine-tune hyperparameters"):
    dt_depth = st.slider("Decision Tree — max depth", 2, 15, 5)
    rf_trees = st.slider("Random Forest — n_estimators", 10, 200, 100, step=10)
    knn_k = st.slider("KNN — k neighbors", 1, 21, 5)

params = dict(dt_depth=dt_depth, rf_trees=rf_trees, knn_k=knn_k)
run_button = st.sidebar.button("⚓ Set Sail — Train Models", type="primary", use_container_width=True)

# --------------------------------------------------------------------------
# HERO HEADER
# --------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero">
        <h1>⚓ Titanic Survival Predictor</h1>
        <p>Interactive ML dashboard comparing Decision Tree, Random Forest, Naive Bayes,
        and KNN to estimate whether a passenger survived — based on class, sex, age,
        fare, and family details aboard.</p>
        <p style="margin-top:10px; font-size:0.85rem;">📂 <b>{data_source}</b>
        {(" · Dropped: " + str(drop_cols)) if drop_cols else ""}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# TOP METRIC CARDS
# --------------------------------------------------------------------------
m1, m2, m3 = st.columns(3)
metric_card(m1, "Total Passengers", f"{len(raw_df):,}")
metric_card(m2, "Features", raw_df.shape[1] - 1)
if target_col in raw_df.columns:
    rate = raw_df[target_col].astype(str).isin(["1", "1.0", "True", "true"]).mean()
    metric_card(m3, "Survival Rate", f"{rate*100:.1f}%")

st.write("")

# --------------------------------------------------------------------------
# DATA + EDA — collapsible sections
# --------------------------------------------------------------------------
with st.expander("🧾 Passenger Manifest (data preview)", expanded=False):
    st.dataframe(raw_df.head(12), use_container_width=True)

with st.expander("🔎 Explore Survival Patterns", expanded=True):
    st.markdown('<span class="section-badge">EXPLORATORY VIEW</span>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        counts = raw_df[target_col].value_counts().rename({0: "Did not survive", 1: "Survived"})
        fig = px.pie(
            names=counts.index, values=counts.values, hole=0.55,
            color_discrete_sequence=["#e07a5f", "#81b29a"],
            title="Outcome Split",
        )
        fig.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        cat_cols = [c for c in raw_df.select_dtypes(include="object").columns if c != target_col]
        num_cols = [c for c in raw_df.select_dtypes(include=np.number).columns if c != target_col]
        feature_choice = st.selectbox("Break down survival by:", cat_cols + num_cols)

        if feature_choice in cat_cols:
            grp = raw_df.groupby(feature_choice)[target_col].apply(
                lambda s: s.astype(str).isin(["1", "1.0", "True", "true"]).mean()
            ).reset_index(name="survival_rate")
            fig2 = px.bar(
                grp, x=feature_choice, y="survival_rate",
                color=feature_choice, color_discrete_sequence=ACCENT_SEQUENCE,
                title=f"Survival Rate by {feature_choice}",
            )
        else:
            fig2 = px.histogram(
                raw_df, x=feature_choice, color=target_col, barmode="overlay",
                color_discrete_sequence=["#e07a5f", "#81b29a"],
                title=f"{feature_choice} Distribution by Outcome",
            )
        fig2.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)", height=340)
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------
# MODEL TRAINING & COMPARISON
# --------------------------------------------------------------------------
st.markdown('<span class="section-badge">MODEL COMPARISON</span>', unsafe_allow_html=True)

if len(selected_algos) == 0:
    st.warning("Pick at least one algorithm from the Control Deck.")
elif not run_button and "results" not in st.session_state:
    st.info("Configure the Control Deck on the left, then click **⚓ Set Sail — Train Models**.")
else:
    if run_button:
        with st.spinner("Training models..."):
            X, y, encoders = preprocess(raw_df, target_col)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            results = {}
            for name in selected_algos:
                cfg = ALGORITHMS[name]
                model = cfg["build"](params)
                xtr = X_train_scaled if cfg["needs_scaling"] else X_train
                xte = X_test_scaled if cfg["needs_scaling"] else X_test
                model.fit(xtr, y_train)
                y_pred = model.predict(xte)
                results[name] = {
                    "model": model, "uses_scaled": cfg["needs_scaling"],
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1": f1_score(y_test, y_pred, zero_division=0),
                    "cm": confusion_matrix(y_test, y_pred),
                }

            st.session_state.update(dict(
                results=results, X_columns=list(X.columns), scaler=scaler,
                encoders=encoders, raw_df_for_predict=raw_df, target_col=target_col,
            ))

    results = st.session_state["results"]
    metrics_df = pd.DataFrame({
        name: {"Accuracy": r["accuracy"], "Precision": r["precision"],
               "Recall": r["recall"], "F1-Score": r["f1"]}
        for name, r in results.items()
    }).T.sort_values("Accuracy", ascending=False)

    best_model = metrics_df.index[0]
    st.success(f"🏆 Top performer: **{best_model}** — {metrics_df.loc[best_model, 'Accuracy']*100:.2f}% accuracy")

    col_table, col_chart = st.columns([1, 1.3])
    with col_table:
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
    with col_chart:
        melted = metrics_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        fig3 = px.bar(
            melted, x="index", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=ACCENT_SEQUENCE,
        )
        fig3.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                            height=340, xaxis_title="", legend_title="")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("**Confusion Matrices**")
    cols = st.columns(len(results))
    for col, (name, r) in zip(cols, results.items()):
        with col:
            fig4 = px.imshow(
                r["cm"], text_auto=True, color_continuous_scale="Blues",
                labels=dict(x="Predicted", y="Actual"),
                x=["No", "Yes"], y=["No", "Yes"],
            )
            fig4.update_layout(template=PLOTLY_TEMPLATE, paper_bgcolor="rgba(0,0,0,0)",
                                height=260, title=name, coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="section-badge">LIVE PREDICTION</span>', unsafe_allow_html=True)
    st.markdown("#### 🔮 Predict for a New Passenger")

    X_cols = st.session_state["X_columns"]
    feature_source = st.session_state["raw_df_for_predict"].drop(columns=[st.session_state["target_col"]])
    model_choice = st.selectbox("Model to use for this prediction", list(results.keys()))

    input_vals = {}
    cols_input = st.columns(3)
    for i, col in enumerate(X_cols):
        series = feature_source[col]
        with cols_input[i % 3]:
            if series.dtype == object:
                input_vals[col] = st.selectbox(col, sorted(series.dropna().unique().tolist()))
            else:
                input_vals[col] = st.number_input(col, value=float(series.median()))

    if st.button("🔮 Predict Survival"):
        input_df = pd.DataFrame([input_vals])
        for col, le in st.session_state["encoders"].items():
            input_df[col] = le.transform(input_df[col].astype(str))
        input_df = input_df[X_cols]

        model_info = results[model_choice]
        input_arr = st.session_state["scaler"].transform(input_df) if model_info["uses_scaled"] else input_df
        pred = model_info["model"].predict(input_arr)[0]
        proba = (
            model_info["model"].predict_proba(input_arr)[0][1]
            if hasattr(model_info["model"], "predict_proba") else None
        )

        if pred == 1:
            st.success(f"✅ Likely to **survive**" + (f" (probability: {proba*100:.1f}%)" if proba is not None else ""))
        else:
            st.error(f"⚠️ Likely to **not survive**" + (f" (probability: {(1-proba)*100:.1f}%)" if proba is not None else ""))

st.markdown("---")
st.caption("Mini Project · Machine Learning · Titanic Survival Predictor · Built with Streamlit & scikit-learn")