"""
data_loader.py
--------------
Handles CSV ingestion into SQLite and feature extraction via SQL queries.
Covers: SQL, Pandas, Data Structures (IQM JD requirements)
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
DB_PATH     = DATA_DIR / "fraud.db"
CSV_PATH    = BASE_DIR / "creditcard.csv"
SAMPLE_PATH = DATA_DIR / "sample_creditcard.csv"


# ── Generate synthetic sample data (used when real CSV is not available) ───────
def generate_sample_data(n_samples: int = 5000, fraud_ratio: float = 0.02) -> pd.DataFrame:
    """
    Generate a synthetic transaction dataset that mirrors the Kaggle credit card
    fraud dataset structure (V1-V28 PCA features + Time + Amount + Class).
    Replace this with the real Kaggle CSV for production use.
    """
    np.random.seed(42)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    def make_rows(n, is_fraud):
        rows = {"Time": np.random.uniform(0, 172800, n)}
        for i in range(1, 29):
            mean = np.random.uniform(-2, 2) if is_fraud else 0
            rows[f"V{i}"] = np.random.normal(mean, 1.5, n)
        rows["Amount"] = (
            np.random.exponential(50, n) if not is_fraud
            else np.random.uniform(1, 2500, n)
        )
        rows["Class"] = int(is_fraud)
        return pd.DataFrame(rows)

    df = pd.concat([make_rows(n_legit, False), make_rows(n_fraud, True)], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── Load CSV → SQLite ──────────────────────────────────────────────────────────
def load_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Persist the DataFrame into a SQLite database and return connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    df.to_sql("transactions", conn, if_exists="replace", index=False)
    print(f"[✓] Loaded {len(df):,} rows into SQLite → {db_path}")
    return conn


# ── SQL Feature Queries ────────────────────────────────────────────────────────
SQL_QUERIES = {
    "hourly_fraud_rate": """
        SELECT
            CAST(Time / 3600 AS INTEGER) AS hour,
            COUNT(*)                      AS total_txns,
            SUM(Class)                    AS fraud_count,
            ROUND(AVG(Class) * 100, 4)    AS fraud_rate_pct,
            ROUND(AVG(Amount), 2)         AS avg_amount
        FROM transactions
        GROUP BY hour
        ORDER BY hour
    """,

    "amount_buckets": """
        SELECT
            CASE
                WHEN Amount < 10   THEN '< $10'
                WHEN Amount < 50   THEN '$10–50'
                WHEN Amount < 200  THEN '$50–200'
                WHEN Amount < 1000 THEN '$200–1k'
                ELSE '> $1k'
            END AS amount_bucket,
            COUNT(*)               AS total,
            SUM(Class)             AS frauds,
            ROUND(AVG(Class)*100, 3) AS fraud_rate_pct
        FROM transactions
        GROUP BY amount_bucket
    """,

    "class_distribution": """
        SELECT
            Class,
            COUNT(*) AS count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 4) AS pct
        FROM transactions
        GROUP BY Class
    """,

    "high_value_fraud": """
        SELECT *
        FROM transactions
        WHERE Class = 1 AND Amount > 1000
        ORDER BY Amount DESC
        LIMIT 20
    """,
}


def run_sql_queries(conn: sqlite3.Connection) -> dict:
    """Execute all analytical SQL queries and return results as DataFrames."""
    results = {}
    for name, query in SQL_QUERIES.items():
        try:
            results[name] = pd.read_sql_query(query, conn)
            print(f"[✓] SQL query '{name}' → {len(results[name])} rows")
        except Exception as e:
            print(f"[✗] SQL query '{name}' failed: {e}")
    return results


# ── Main data loading pipeline ─────────────────────────────────────────────────
def load_data():
    """
    Full data loading pipeline:
    1. Read CSV (or generate synthetic data)
    2. Load into SQLite
    3. Run SQL feature queries
    Returns: (raw DataFrame, DB connection, SQL results dict)
    """
    # Try real Kaggle CSV first, fall back to synthetic data
    if CSV_PATH.exists():
        print(f"[✓] Loading real dataset from {CSV_PATH}")
        df = pd.read_csv(str(CSV_PATH))
    elif SAMPLE_PATH.exists():
        print(f"[!] Using sample dataset from {SAMPLE_PATH}")
        df = pd.read_csv(str(SAMPLE_PATH))
    else:
        print("[!] creditcard.csv not found — generating synthetic data (5,000 rows)")
        print("    → Download real data: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        df = generate_sample_data(n_samples=5000)
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(str(SAMPLE_PATH), index=False)
        print(f"[✓] Saved synthetic sample to {SAMPLE_PATH}")

    # Basic validation
    required_cols = {"Time", "Amount", "Class"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    print(f"\n📊 Dataset Summary:")
    print(f"   Rows     : {len(df):,}")
    print(f"   Columns  : {len(df.columns)}")
    print(f"   Frauds   : {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"   Nulls    : {df.isnull().sum().sum()}")

    conn = load_to_sqlite(df)
    sql_results = run_sql_queries(conn)

    return df, conn, sql_results


if __name__ == "__main__":
    df, conn, results = load_data()
    print("\n── SQL Results Preview ──")
    for name, result_df in results.items():
        print(f"\n{name}:")
        print(result_df.to_string(index=False))
    conn.close()
