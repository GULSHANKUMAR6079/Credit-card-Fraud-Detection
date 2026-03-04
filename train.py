"""
train.py
--------
Train, evaluate, and save multiple ML models.
Covers: Sklearn, XGBoost, model comparison, Pipeline API (IQM JD)
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)



# ── Model Definitions ──────────────────────────────────────────────────────────
def get_models(preprocessor) -> dict:
    """Return dict of named Sklearn Pipelines."""
    models = {}

    # Logistic Regression (baseline)
    models["logistic_regression"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, C=0.1, random_state=42
        )),
    ])

    # Random Forest
    models["random_forest"] = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200, max_depth=15,
            class_weight="balanced", n_jobs=-1, random_state=42
        )),
    ])

    # XGBoost
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=100, eval_metric="aucpr",
                use_label_encoder=False, random_state=42, verbosity=0
            )),
        ])
    except ImportError:
        print("[!] XGBoost not installed — skipping")

    return models


# ── Evaluate a single model ────────────────────────────────────────────────────
def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute all relevant metrics for a trained model."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        "name":             name,
        "roc_auc":          round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision":    round(average_precision_score(y_test, y_prob), 4),
        "precision":        round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["precision"]), 4),
        "recall":           round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["recall"]), 4),
        "f1":               round(float(classification_report(y_test, y_pred, output_dict=True)["1"]["f1-score"]), 4),
    }

    print(f"\n── {name} ──────────────────────────")
    print(f"   ROC-AUC       : {metrics['roc_auc']}")
    print(f"   Avg Precision : {metrics['avg_precision']}")
    print(f"   Precision     : {metrics['precision']}")
    print(f"   Recall        : {metrics['recall']}")
    print(f"   F1 Score      : {metrics['f1']}")

    return metrics, y_prob, y_pred


# ── Isolation Forest (unsupervised) ───────────────────────────────────────────
def train_isolation_forest(X_train, X_test, y_test, preprocessor) -> dict:
    """Train Isolation Forest for anomaly-based fraud detection."""
    from sklearn.preprocessing import FunctionTransformer

    # Fit preprocessor first
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    iso = IsolationForest(contamination=0.02, n_estimators=200,
                          n_jobs=-1, random_state=42)
    iso.fit(X_train_t)

    # Isolation Forest returns -1 (anomaly) or 1 (normal)
    scores = -iso.decision_function(X_test_t)   # higher = more anomalous
    preds  = (iso.predict(X_test_t) == -1).astype(int)

    metrics = {
        "name":          "isolation_forest",
        "roc_auc":       round(roc_auc_score(y_test, scores), 4),
        "avg_precision": round(average_precision_score(y_test, scores), 4),
        "precision":     round(float(classification_report(y_test, preds, output_dict=True, zero_division=0)["1"]["precision"]), 4),
        "recall":        round(float(classification_report(y_test, preds, output_dict=True, zero_division=0)["1"]["recall"]), 4),
        "f1":            round(float(classification_report(y_test, preds, output_dict=True, zero_division=0)["1"]["f1-score"]), 4),
    }

    print(f"\n── isolation_forest ───────────────")
    for k, v in metrics.items():
        if k != "name":
            print(f"   {k:<20}: {v}")

    return metrics, scores, preds, iso


# ── Plot: ROC Curves ──────────────────────────────────────────────────────────
def plot_roc_curves(results_list: list, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")

    colors = ["#00ff88", "#3b82f6", "#ff3366", "#f59e0b", "#a78bfa"]

    for i, (name, y_test, y_prob) in enumerate(results_list):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate", color="#94a3b8")
    ax.set_ylabel("True Positive Rate", color="#94a3b8")
    ax.set_title("ROC Curves — Fraud Detection Models", color="#e2e8f0", fontsize=13)
    ax.legend(loc="lower right", facecolor="#1e1e2e", labelcolor="#e2e8f0")
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] ROC curve saved → {save_path}")


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(name: str, y_test, y_pred, save_path: Path):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")

    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"],
                ax=ax, cbar=False)
    ax.set_title(f"Confusion Matrix — {name}", color="#e2e8f0")
    ax.set_xlabel("Predicted", color="#94a3b8")
    ax.set_ylabel("Actual", color="#94a3b8")
    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


# ── Main Training Pipeline ────────────────────────────────────────────────────
def train_all(X_train, X_test, y_train, y_test, feature_cols):
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from feature_engineering import build_preprocessor

    preprocessor = build_preprocessor(list(X_train.columns))
    models       = get_models(preprocessor)

    all_metrics  = []
    roc_data     = []
    best_model   = None
    best_auc     = 0

    # ── Train supervised models ────────────────────────────────────────────────
    for name, pipeline in models.items():
        print(f"\n[⚙] Training {name}...")
        pipeline.fit(X_train, y_train)
        metrics, y_prob, y_pred = evaluate_model(name, pipeline, X_test, y_test)
        all_metrics.append(metrics)
        roc_data.append((name, y_test, y_prob))

        # Plot confusion matrix per model
        plot_confusion_matrix(
            name, y_test, y_pred,
            REPORTS_DIR / f"confusion_{name}.png"
        )

        # Save model
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(pipeline, str(model_path))
        print(f"[✓] Saved → {model_path}")

        if metrics["roc_auc"] > best_auc:
            best_auc   = metrics["roc_auc"]
            best_model = (name, pipeline)

    # ── Train Isolation Forest ─────────────────────────────────────────────────
    from feature_engineering import build_preprocessor as bp2
    iso_prep    = bp2(list(X_train.columns))
    iso_metrics, iso_scores, iso_preds, iso_model = train_isolation_forest(
        X_train, X_test, y_test, iso_prep
    )
    all_metrics.append(iso_metrics)
    roc_data.append(("isolation_forest", y_test, iso_scores))
    joblib.dump(iso_model, str(MODELS_DIR / "isolation_forest.joblib"))

    # ── ROC Curve plot ─────────────────────────────────────────────────────────
    plot_roc_curves(roc_data, REPORTS_DIR / "roc_curves.png")

    # ── Metrics summary ────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(str(REPORTS_DIR / "model_metrics.csv"), index=False)
    print(f"\n\n{'='*50}")
    print("MODEL COMPARISON SUMMARY")
    print('='*50)
    print(metrics_df.to_string(index=False))
    print(f"\n🏆 Best Model: {best_model[0]} (AUC={best_auc:.4f})")

    # Save best model reference
    if best_model:
        joblib.dump(best_model[1], str(MODELS_DIR / "best_model.joblib"))
        with open(str(MODELS_DIR / "best_model_name.txt"), "w") as f:
            f.write(best_model[0])
        print(f"[✓] Best model saved → models/best_model.joblib")

    return all_metrics, best_model


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from data_loader import load_data
    from feature_engineering import engineer_features, prepare_datasets

    print("=" * 50)
    print("FRAUD DETECTION — MODEL TRAINING")
    print("=" * 50)

    df, conn, _ = load_data()
    conn.close()

    df_feat = engineer_features(df)
    X_train, X_test, y_train, y_test, features = prepare_datasets(df_feat)

    train_all(X_train, X_test, y_train, y_test, features)
