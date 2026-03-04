"""
online_pipeline_demo.py
-----------------------
Online / Streaming ML simulation — works without River installed.
Uses sklearn's SGDClassifier which supports partial_fit() for incremental learning.
Install River for the full version: pip install river
"""

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def run_online_sgd(df: pd.DataFrame, n_samples: int = 3000, report_every: int = 100):
    """
    Incremental online learning using SGDClassifier.partial_fit().
    Simulates real-time fraud detection with prequential evaluation.
    """
    feature_cols = [c for c in df.columns if c not in ("Class", "Time")]
    df_sample    = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    X = df_sample[feature_cols].fillna(0).values
    y = df_sample["Class"].values

    # Online scaler (update stats incrementally)
    scaler  = StandardScaler()
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    sample_weight_map = {0: class_weights[0], 1: class_weights[1]}
    clf = SGDClassifier(loss="log_loss", learning_rate="optimal", eta0=0.01, random_state=42)

    history_auc   = []
    history_steps = []
    y_true_window = deque(maxlen=500)
    y_prob_window = deque(maxlen=500)
    fraud_caught  = 0

    print(f"[⚙] Running online SGD on {len(df_sample):,} transactions...")
    start = time.time()

    # Warm-start: fit scaler + classifier on first small batch
    warm_size = 50
    scaler.partial_fit(X[:warm_size])
    sw = [sample_weight_map[yi] for yi in y[:warm_size]]
    clf.partial_fit(X[:warm_size], y[:warm_size], classes=[0, 1], sample_weight=sw)

    for i in range(warm_size, len(X)):
        xi = X[i].reshape(1, -1)
        yi = y[i]

        # Scale
        xi_scaled = scaler.transform(xi)

        # Predict (prequential: predict before learning)
        prob = clf.predict_proba(xi_scaled)[0][1]
        pred = int(prob >= 0.5)

        y_true_window.append(yi)
        y_prob_window.append(prob)

        if yi == 1 and pred == 1:
            fraud_caught += 1

        # Learn from this sample
        scaler.partial_fit(xi)
        clf.partial_fit(xi_scaled, [yi], classes=[0, 1], sample_weight=[sample_weight_map[yi]])

        # Report periodically
        if (i + 1) % report_every == 0 and len(set(y_true_window)) > 1:
            auc = roc_auc_score(list(y_true_window), list(y_prob_window))
            history_auc.append(auc)
            history_steps.append(i + 1)
            elapsed = time.time() - start
            print(f"   Step {i+1:>5} | AUC: {auc:.4f} | Fraud caught: {fraud_caught} | {elapsed:.1f}s")

    final_auc = history_auc[-1] if history_auc else 0.0
    print(f"\n[✓] Online SGD complete")
    print(f"    Final AUC   : {final_auc:.4f}")
    print(f"    Fraud caught: {fraud_caught}/{int(y.sum())}")

    # Plot learning curve
    if history_steps:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#0a0a0f")
        ax.set_facecolor("#111118")
        ax.plot(history_steps, history_auc, color="#00ff88", linewidth=2.5)
        ax.fill_between(history_steps, history_auc, alpha=0.15, color="#00ff88")
        ax.set_xlabel("Transactions Processed", color="#94a3b8")
        ax.set_ylabel("ROC-AUC (rolling 500)", color="#94a3b8")
        ax.set_title("Online Learning — SGD Classifier (AUC over time)",
                     color="#e2e8f0", fontsize=13)
        ax.tick_params(colors="#64748b")
        ax.grid(alpha=0.1, color="#1e1e2e")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e1e2e")
        plt.tight_layout()
        save_path = REPORTS_DIR / "online_learning_curve.png"
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
        plt.close()
        print(f"[✓] Online learning curve saved → {save_path}")

    return {"final_auc": round(final_auc, 4), "fraud_caught": fraud_caught}


def compare_offline_vs_online(offline_auc: float, online_auc: float):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0a0a0f")
    ax.set_facecolor("#111118")

    models = ["Offline\n(Batch RF)", "Online\n(SGD Stream)"]
    aucs   = [offline_auc, online_auc]
    colors = ["#00ff88", "#3b82f6"]
    bars   = ax.bar(models, aucs, color=colors, width=0.4, edgecolor="#1e1e2e")

    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", color="#e2e8f0", fontsize=13, fontweight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("ROC-AUC Score", color="#94a3b8")
    ax.set_title("Offline Batch vs Online Streaming — AUC Comparison",
                 color="#e2e8f0", fontsize=13)
    ax.tick_params(colors="#94a3b8")
    ax.grid(axis="y", alpha=0.1, color="#1e1e2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e1e2e")

    plt.tight_layout()
    save_path = REPORTS_DIR / "offline_vs_online.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight", facecolor="#0a0a0f")
    plt.close()
    print(f"[✓] Comparison chart saved → {save_path}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from data_loader import load_data
    from feature_engineering import engineer_features

    print("=" * 50)
    print("ONLINE STREAMING FRAUD DETECTION — SGD")
    print("=" * 50)

    df, conn, _ = load_data()
    conn.close()
    df_feat = engineer_features(df)

    results = run_online_sgd(df_feat, n_samples=3000, report_every=100)

    # Load best offline AUC from metrics CSV
    metrics_path = BASE_DIR / "reports" / "model_metrics.csv"
    if metrics_path.exists():
        offline_auc = pd.read_csv(str(metrics_path))["roc_auc"].max()
    else:
        offline_auc = 0.99

    compare_offline_vs_online(offline_auc, results["final_auc"])
