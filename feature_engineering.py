"""
feature_engineering.py
-----------------------
Feature engineering pipeline using Pandas + Sklearn.
Creates time-based, amount-based, and statistical features.
Handles class imbalance with SMOTE.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


# ── Raw Feature Engineering (Pandas) ──────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from raw transaction data.
    - Time decomposition
    - Amount transformations
    - Statistical aggregations
    """
    df = df.copy()

    # ── Time features ──────────────────────────────────────────────────────────
    df["hour"]       = (df["Time"] // 3600).astype(int) % 24   # hour of day 0-23
    df["day"]        = (df["Time"] // 86400).astype(int)        # day number
    df["is_night"]   = ((df["hour"] >= 22) | (df["hour"] <= 5)).astype(int)
    df["is_weekend"] = (df["day"] % 7 >= 5).astype(int)

    # ── Amount features ────────────────────────────────────────────────────────
    df["log_amount"]    = np.log1p(df["Amount"])               # log-transform for skew
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    df["is_round"]      = (df["Amount"] % 10 == 0).astype(int) # round amounts = suspicious

    # ── Rolling stats (simulating per-user velocity features) ─────────────────
    # Using global rolling as proxy (real system would group by user_id)
    df_sorted = df.sort_values("Time").copy()
    df_sorted["rolling_mean_amt"] = (
        df_sorted["Amount"].rolling(window=50, min_periods=1).mean()
    )
    df_sorted["rolling_std_amt"] = (
        df_sorted["Amount"].rolling(window=50, min_periods=1).std().fillna(0)
    )
    df_sorted["amount_vs_rolling"] = (
        df_sorted["Amount"] / (df_sorted["rolling_mean_amt"] + 1e-9)
    )

    # Re-align with original index
    df["rolling_mean_amt"]    = df_sorted["rolling_mean_amt"]
    df["rolling_std_amt"]     = df_sorted["rolling_std_amt"]
    df["amount_vs_rolling"]   = df_sorted["amount_vs_rolling"]

    # ── Interaction features ───────────────────────────────────────────────────
    df["night_high_amount"] = df["is_night"] * (df["Amount"] > 500).astype(int)
    df["v1_v2_interaction"] = df["V1"] * df["V2"]

    print(f"[✓] Feature engineering complete — {len(df.columns)} total columns")
    return df


# ── Build Sklearn Preprocessing Pipeline ──────────────────────────────────────
def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """
    Sklearn ColumnTransformer:
    - RobustScaler for amount-based features (handles outliers)
    - StandardScaler for PCA and engineered features
    """
    amount_cols = ["Amount", "log_amount", "amount_zscore",
                   "rolling_mean_amt", "rolling_std_amt", "amount_vs_rolling"]
    binary_cols = ["is_night", "is_weekend", "is_round", "night_high_amount"]
    amount_cols  = [c for c in amount_cols if c in feature_cols]
    binary_cols  = [c for c in binary_cols if c in feature_cols]
    other_cols   = [c for c in feature_cols
                    if c not in amount_cols and c not in binary_cols]

    preprocessor = ColumnTransformer(transformers=[
        ("robust",   RobustScaler(),   amount_cols),
        ("standard", StandardScaler(), other_cols),
        ("passthrough", "passthrough", binary_cols),
    ], remainder="drop")

    return preprocessor


# ── Train/Test Split with SMOTE ────────────────────────────────────────────────
def prepare_datasets(df: pd.DataFrame,
                     test_size: float = 0.2,
                     use_smote: bool = True):
    """
    Split data, optionally apply SMOTE to handle class imbalance.
    Returns: X_train, X_test, y_train, y_test, feature_cols
    """
    drop_cols = ["Class", "Time"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["Class"]

    # Stratified split to preserve fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print(f"[✓] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"    Train fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"    Test  fraud: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")

    # SMOTE oversampling on training set only
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=42, k_neighbors=5)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print(f"[✓] SMOTE applied — train size: {len(X_train_res):,} "
                  f"(fraud: {y_train_res.sum():,})")
            return X_train_res, X_test, y_train_res, y_test, feature_cols
        except ImportError:
            print("[!] imbalanced-learn not installed — skipping SMOTE")

    return X_train, X_test, y_train, y_test, feature_cols


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(BASE_DIR / "src"))
    from data_loader import load_data

    df, conn, _ = load_data()
    conn.close()

    df_feat = engineer_features(df)
    X_train, X_test, y_train, y_test, features = prepare_datasets(df_feat)

    print(f"\n[✓] Features used ({len(features)}):")
    for f in features:
        print(f"    {f}")
