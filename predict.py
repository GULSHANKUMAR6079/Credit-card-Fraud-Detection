"""
predict.py
----------
Inference module: load saved model and predict on new transactions.
Also includes SHAP explainability.
"""

import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# ── Load saved model ───────────────────────────────────────────────────────────
def load_model(model_name: str = "best_model"):
    model_path = MODELS_DIR / f"{model_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run train.py first."
        )
    model = joblib.load(str(model_path))
    print(f"[✓] Loaded model: {model_name}")
    return model


# ── Single transaction prediction ─────────────────────────────────────────────
def predict_transaction(transaction: dict, model) -> dict:
    """
    Predict fraud probability for a single transaction.
    Input: dict with keys matching training features
    Output: dict with prediction + probability + risk_level
    """
    df = pd.DataFrame([transaction])

    # Add engineered features if missing
    if "log_amount" not in df.columns and "Amount" in df.columns:
        df["log_amount"]    = np.log1p(df["Amount"])
        df["amount_zscore"] = 0.0   # unknown global stats at inference time
        df["is_round"]      = (df["Amount"] % 10 == 0).astype(int)

    if "hour" not in df.columns and "Time" in df.columns:
        df["hour"]       = (df["Time"] // 3600).astype(int) % 24
        df["day"]        = (df["Time"] // 86400).astype(int)
        df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
        df["is_weekend"] = (df["day"] % 7 >= 5).astype(int)

    # Fill any remaining missing cols with 0
    try:
        fraud_prob = model.predict_proba(df)[0][1]
    except Exception as e:
        return {"error": str(e), "fraud_probability": None}

    risk_level = (
        "HIGH"   if fraud_prob > 0.7 else
        "MEDIUM" if fraud_prob > 0.3 else
        "LOW"
    )

    return {
        "fraud_probability": round(float(fraud_prob), 4),
        "is_fraud":          int(fraud_prob > 0.5),
        "risk_level":        risk_level,
        "action":            "🚨 BLOCK" if risk_level == "HIGH" else
                             "⚠️  REVIEW" if risk_level == "MEDIUM" else
                             "✅ APPROVE",
    }


# ── Batch prediction ───────────────────────────────────────────────────────────
def predict_batch(df: pd.DataFrame, model) -> pd.DataFrame:
    """Run predictions on a DataFrame of transactions."""
    probs  = model.predict_proba(df)[:, 1]
    preds  = (probs > 0.5).astype(int)
    risk   = pd.cut(probs, bins=[0, 0.3, 0.7, 1.0],
                    labels=["LOW", "MEDIUM", "HIGH"])

    result = df.copy()
    result["fraud_probability"] = probs.round(4)
    result["predicted_fraud"]   = preds
    result["risk_level"]        = risk
    return result


# ── SHAP Explainability ────────────────────────────────────────────────────────
def explain_with_shap(model, X_sample: pd.DataFrame, max_display: int = 15):
    """
    Generate SHAP summary plot to explain model predictions.
    Shows which features most influence fraud detection.
    """
    try:
        import shap
    except ImportError:
        print("[!] SHAP not installed. Run: pip install shap")
        return

    print("[⚙] Computing SHAP values (this may take a moment)...")

    # Get the classifier from the pipeline
    clf = model.named_steps["classifier"]
    pre = model.named_steps["preprocessor"]

    X_transformed = pre.transform(X_sample)

    # Use TreeExplainer for tree-based models, else LinearExplainer
    try:
        explainer    = shap.TreeExplainer(clf)
        shap_values  = explainer.shap_values(X_transformed)
        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        explainer   = shap.LinearExplainer(clf, X_transformed)
        shap_values = explainer.shap_values(X_transformed)

    # Feature names after transformation
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"feat_{i}" for i in range(X_transformed.shape[1])]

    # ── SHAP Summary Plot ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")

    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        max_display=max_display,
        show=False, plot_type="bar"
    )

    plt.title("SHAP Feature Importance — Fraud Detection",
              color="#e2e8f0", fontsize=13, pad=15)
    plt.tight_layout()
    save_path = REPORTS_DIR / "shap_summary.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight",
                facecolor="#0a0a0f")
    plt.close()
    print(f"[✓] SHAP summary saved → {save_path}")
    return shap_values, feature_names


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from data_loader import load_data
    from feature_engineering import engineer_features, prepare_datasets

    # Load data & model
    df, conn, _ = load_data()
    conn.close()
    df_feat = engineer_features(df)
    X_train, X_test, y_train, y_test, features = prepare_datasets(df_feat, use_smote=False)

    try:
        model = load_model("best_model")
    except FileNotFoundError:
        print("[!] No trained model found. Run train.py first.")
        exit(1)

    # ── Demo: single transaction prediction ────────────────────────────────────
    print("\n── Single Transaction Prediction Demo ──")
    sample_txn = X_test.iloc[0].to_dict()
    result = predict_transaction(sample_txn, model)
    print(f"   Fraud Probability : {result['fraud_probability']}")
    print(f"   Risk Level        : {result['risk_level']}")
    print(f"   Action            : {result['action']}")

    # ── Demo: batch prediction ─────────────────────────────────────────────────
    print("\n── Batch Prediction (first 100 test samples) ──")
    batch_results = predict_batch(X_test.head(100), model)
    fraud_found   = batch_results["predicted_fraud"].sum()
    print(f"   Transactions scanned : 100")
    print(f"   Fraud detected       : {fraud_found}")

    # ── SHAP explanation ───────────────────────────────────────────────────────
    print("\n── Generating SHAP Explanations ──")
    explain_with_shap(model, X_test.head(200))
