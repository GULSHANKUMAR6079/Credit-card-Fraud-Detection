# 🔍 Real-Time Credit Card Fraud Detection

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

A production-grade end-to-end fraud detection pipeline covering **offline batch ML** and **online streaming ML** with a live Streamlit dashboard — built for IQM Corporation's Data Scientist interview.

---

## 🚀 Live Demo

> **[→ Launch Dashboard](https://share.streamlit.io)**  
> *(Replace with your Streamlit Community Cloud URL after deployment)*

---

## 📁 Project Structure

```
Credit Card Fraud Detection/
├── app.py                  ← Streamlit dashboard (5 pages)
├── data_loader.py          ← CSV ingestion + SQLite + SQL queries
├── feature_engineering.py  ← Feature creation + SMOTE preprocessing
├── train.py                ← Train & evaluate 4 ML models
├── predict.py              ← Inference on new transactions
├── online_pipeline_demo.py ← River streaming model demo
├── requirements.txt
├── data/                   ← SQLite DB (auto-generated)
├── models/                 ← Saved .joblib models (auto-generated)
└── reports/                ← Charts & metrics (auto-generated)
```

---

## ⚡ Quick Start (Local)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection
cd "credit-card-fraud-detection"

# 2. Create virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt

# 3. (Optional) Add the real Kaggle dataset for full 284k rows
#    Download: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    Place creditcard.csv in the project root
#    Without it, the app auto-generates 5,000 synthetic rows for preview

# 4. (Optional) Train models to generate reports & model files
python train.py

# 5. Launch the dashboard
streamlit run app.py
```

---

## 🤖 Models Used

| Model | Type | Library | AUC-ROC |
|-------|------|---------|---------|
| XGBoost | Supervised | xgboost | ~0.98 |
| Random Forest | Supervised | sklearn | ~0.97 |
| Logistic Regression | Supervised | sklearn | ~0.95 |
| Isolation Forest | Unsupervised | sklearn | ~0.75 |
| Hoeffding Tree | Online/Streaming | river | ~0.91 |

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 Overview | KPI cards, fraud rate, amount distribution, hourly patterns |
| 🤖 Model Comparison | ROC-AUC bar charts, precision vs recall scatter, metrics table |
| 🔎 Transaction Analyzer | Real-time risk scoring with interactive gauge chart |
| 📈 Trend Analysis | Box plots, night vs day fraud rate, feature correlations |
| ⚡ Online Pipeline | River streaming ML explanation & live code demo |

---

## 🎯 IQM JD Coverage

- ✅ Python + Pandas + Scikit-learn
- ✅ SQL (SQLite) for data ingestion & querying
- ✅ Offline + Online ML (batch + streaming, both implemented)
- ✅ Stakeholder dashboard (Streamlit + Plotly)
- ✅ Class imbalance handling (SMOTE)
- ✅ Large dataset (284k transactions, 0.17% fraud)
- ✅ Modular, production-ready architecture

---

## 📋 Dataset

**Kaggle Credit Card Fraud Detection**

- 284,807 transactions | 492 fraud cases (0.17%)
- Features: V1–V28 (PCA-anonymized) + Time + Amount + Class
- 🔗 [Download here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

> **Note:** The CSV is not included in this repo (150 MB). The app automatically uses synthetic preview data if the CSV is not present.

---

## 👤 Author

Built for **IQM Corporation Data Science Interview, 2026**
