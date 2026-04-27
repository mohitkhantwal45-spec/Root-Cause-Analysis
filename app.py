import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, f1_score
)
from imblearn.over_sampling import SMOTE
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Steel Plate Defect – Root Cause Analysis",
    page_icon="🔩",
    layout="wide",
)

CLASS_NAMES = ['Bumps', 'Dirtiness', 'K_Scratch', 'Other_Faults',
               'Pastry', 'Stains', 'Z_Scratch']

# ── Data & model loading (cached) ──────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching dataset from UCI…")
def load_data():
    steel_plates = fetch_ucirepo(id=198)
    X = steel_plates.data.features
    y_multi = steel_plates.data.targets
    y_raw = y_multi.idxmax(axis=1)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le


@st.cache_resource(show_spinner="Training models (this takes ~1–2 min)…")
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), X.columns.tolist())
    ])

    def make_pipe(clf):
        return Pipeline([('preprocessor', preprocessor), ('classifier', clf)])

    lr = make_pipe(LogisticRegression(multi_class='multinomial', solver='lbfgs',
                                      max_iter=3000, random_state=42))
    rf = make_pipe(RandomForestClassifier(n_estimators=400, max_depth=15,
                                          random_state=42, n_jobs=-1))
    xgb = make_pipe(XGBClassifier(objective='multi:softprob', num_class=7,
                                   max_depth=8, learning_rate=0.08,
                                   n_estimators=400, subsample=0.85,
                                   colsample_bytree=0.8, random_state=42,
                                   n_jobs=-1, eval_metric='mlogloss',
                                   verbosity=0))

    lr.fit(X_train_s, y_train_s)
    rf.fit(X_train_s, y_train_s)
    xgb.fit(X_train_s, y_train_s)

    return lr, rf, xgb, X_train, X_test, y_train, y_test, X_train_s, y_train_s


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_metrics(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred)
    return dict(acc=acc, macro_f1=macro_f1,
                precision=precision, recall=recall, f1=f1,
                support=support, cm=cm, y_pred=y_pred)


# ── App ────────────────────────────────────────────────────────────────────────
st.title("🔩 Steel Plate Defect — Root Cause Analysis")
st.markdown(
    "This app trains **Logistic Regression**, **Random Forest**, and **XGBoost** "
    "on the UCI Steel Plates Faults dataset to classify 7 defect types and identify "
    "the features most responsible for each defect."
)

# ── Load data ──────────────────────────────────────────────────────────────────
X, y, le = load_data()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox(
        "Active model for deep-dive",
        ["Logistic Regression", "Random Forest", "XGBoost"],
        index=2,
    )
    top_n = st.slider("Top N features (importance chart)", 5, 27, 15)
    st.markdown("---")
    st.caption("Data: UCI ML Repo — Steel Plates Faults (ID 198)")

# ── Dataset Overview ───────────────────────────────────────────────────────────
with st.expander("📊 Dataset Overview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", X.shape[0])
    col2.metric("Features", X.shape[1])
    col3.metric("Defect Classes", 7)

    fig, ax = plt.subplots(figsize=(9, 3))
    pd.Series(y).map(dict(enumerate(CLASS_NAMES))).value_counts().plot(
        kind='bar', color='steelblue', ax=ax
    )
    ax.set_title("Class Distribution")
    ax.set_xlabel("Defect Type")
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Train ──────────────────────────────────────────────────────────────────────
lr, rf, xgb, X_train, X_test, y_train, y_test, X_train_s, y_train_s = train_models(X, y)

pipe_map = {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb}
active_pipe = pipe_map[selected_model]

# ── Model Comparison ───────────────────────────────────────────────────────────
st.subheader("📈 Model Comparison")

rows = []
for name, pipe in pipe_map.items():
    m = get_metrics(pipe, X_test, y_test)
    rows.append({"Model": name, "Accuracy": round(m['acc'], 4), "Macro F1": round(m['macro_f1'], 4)})

df_compare = pd.DataFrame(rows).set_index("Model")
st.dataframe(df_compare.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

# ── Deep-dive: selected model ──────────────────────────────────────────────────
st.subheader(f"🔍 Deep-dive: {selected_model}")

m = get_metrics(active_pipe, X_test, y_test)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Accuracy", f"{m['acc']:.4f}")
col_b.metric("Macro Precision", f"{m['precision'].mean():.4f}")
col_c.metric("Macro F1", f"{m['macro_f1']:.4f}")

# Per-class metrics bar chart
tab1, tab2 = st.tabs(["Per-class Metrics", "Confusion Matrix"])

with tab1:
    df_cls = pd.DataFrame({
        'Defect': CLASS_NAMES,
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1-Score': m['f1'],
    }).set_index('Defect')
    fig, ax = plt.subplots(figsize=(10, 4))
    df_cls.plot(kind='bar', ax=ax, width=0.75)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-class Metrics — {selected_model}")
    ax.set_ylabel("Score")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab2:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(m['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix — {selected_model}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Actual vs Predicted ────────────────────────────────────────────────────────
st.subheader("📉 Actual vs Predicted Counts")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, (name, pipe) in zip(axes, pipe_map.items()):
    y_p = pipe.predict(X_test)
    act = pd.Series(y_test).value_counts().sort_index()
    pred = pd.Series(y_p).value_counts().sort_index()
    ax.plot(CLASS_NAMES, act.values, marker='o', label='Actual', color='steelblue')
    ax.plot(CLASS_NAMES, pred.values, marker='s', label='Predicted', color='darkorange')
    ax.set_title(name)
    ax.set_xlabel("Defect Type")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.suptitle("Actual vs Predicted — All Models", fontsize=14, y=1.02)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Root Cause: Feature Importance ────────────────────────────────────────────
st.subheader("🌳 Root Cause Analysis — Feature Importance")

if selected_model in ("Random Forest", "XGBoost"):
    importances = active_pipe.named_steps['classifier'].feature_importances_
    fi_df = (pd.DataFrame({'Feature': X.columns, 'Importance': importances})
             .sort_values('Importance', ascending=False)
             .head(top_n))
    fig, ax = plt.subplots(figsize=(10, top_n * 0.45 + 1))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title(f"Top {top_n} Features — {selected_model}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)
else:
    st.info("Feature importance is available for Random Forest and XGBoost. "
            "Switch the model in the sidebar to view it.")

# ── Predict on custom input ────────────────────────────────────────────────────
st.subheader("🧪 Predict a Custom Sample")
st.markdown("Adjust feature values and get a real-time defect prediction.")

with st.form("predict_form"):
    defaults = X.median().to_dict()
    cols = st.columns(4)
    user_input = {}
    for i, col in enumerate(X.columns):
        c = cols[i % 4]
        user_input[col] = c.number_input(col, value=float(defaults[col]), format="%.4f")
    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    sample = pd.DataFrame([user_input])
    pred_idx = active_pipe.predict(sample)[0]
    proba = active_pipe.predict_proba(sample)[0]
    pred_label = CLASS_NAMES[pred_idx]

    st.success(f"**Predicted Defect:** {pred_label}")
    prob_df = pd.DataFrame({'Defect': CLASS_NAMES, 'Probability': proba}).sort_values('Probability', ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(data=prob_df, x='Defect', y='Probability', palette='Blues_d', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
