# 🏢 American Companies Bankruptcy Prediction System

[![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.0%2B-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-5.0%2B-2A4B7C?logo=optuna&logoColor=white)](https://optuna.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black%20%2F%20PEP8-000000.svg)](https://pep8.org/)

An enterprise-grade, end-to-end Machine Learning system that predicts corporate bankruptcy risk for American companies using fundamental accounting indicators and balance sheet metrics. Designed with modular pipeline architecture, Bayesian hyperparameter optimization, group-aware cross-validation, precision-recall threshold calibration, an interactive Streamlit UI, and a high-performance FastAPI backend containerized with Docker.

---



## 📌 Overview & Problem Statement

Corporate bankruptcy prediction is critical for financial institutions, credit rating agencies, lenders, and investors to assess default probability and mitigate credit risk. 

This project uses historical financial statements from American public companies (dataset: [`utkarshx27/american-companies-bankruptcy-prediction-dataset`](https://www.kaggle.com/datasets/utkarshx27/american-companies-bankruptcy-prediction-dataset)). It provides an automated, leak-free machine learning workflow:
1. **Automated Ingestion & Validation**: Downloads raw data from Kaggle and verifies schema integrity.
2. **Domain-Specific Feature Engineering**: Normalizes financial metrics by total assets, applies logarithmic scaling to asset values, and mitigates financial outliers via percentile clipping.
3. **Multi-Model Tournament**: Evaluates 7 distinct classification algorithms with Optuna Bayesian search under group-aware cross-validation (`GroupKFold` grouped by company).
4. **Threshold Calibration**: Adjusts decision boundaries on out-of-fold predictions to maximize F1-score on severe class imbalance.
5. **Full-Stack Deployment**: Serves predictions via a FastAPI microservice consumed by a modern Streamlit dashboard, both orchestrated via Docker Compose.

---

## ⚡ Key Architectural Highlights

- **Leak-Free Temporal Splitting**: The dataset is split temporally (`year < 2012` for training, `year >= 2012` for testing) to prevent look-ahead bias and simulate real-world financial forecasting.
- **GroupKFold Cross-Validation**: Validation folds are grouped by `company_name`, ensuring multiple fiscal years from the same company never appear in both training and validation folds simultaneously.
- **Asset Ratio Transformation**: Raw financial indicators ($X_1 - X_{18}$) are standardized by dividing by Total Assets ($\frac{X_i}{\text{Total Assets} + \epsilon}$), combined with $\log(1 + \text{Total Assets})$ and 1st/99th percentile quantile clipping.
- **Optuna Hyperparameter Tuning**: Automated search over 7 model families:
  - **CatBoost** (Current Best Model)
  - **LightGBM**
  - **XGBoost**
  - **Random Forest**
  - **Extra Trees**
  - **HistGradientBoosting**
  - **Logistic Regression**
- **Optimal Threshold Calibration**: Rather than defaulting to $0.5$, the decision threshold is calibrated using the precision-recall curve on out-of-fold validation probabilities to optimize detection of rare bankruptcy events.
- **Experiment Tracking**: Comprehensive logging of hyperparameters, CV scores, ROC curves, PR curves, feature importances, and confusion matrices into MLflow.
- **Production-Ready Serving**: FastAPI endpoint validated with Pydantic schemas, wrapped in Docker containers with integrated health checks.

---

## 📈 Model Performance & Evaluation

The system was evaluated on the out-of-time test dataset (`year >= 2012`), achieving high discrimination and robust recall for default identification under severe class imbalance:

| Metric | Score | Key Takeaway |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8419** | Strong ranking ability to distinguish bankrupt from non-bankrupt entities |
| **Recall (Sensitivity)** | **71.34%** | Captures over 71% of all corporate failures in advance |
| **Specificity** | **81.44%** | Correctly identifies ~81.4% of solvent/healthy businesses |
| **Accuracy** | **81.10%** | Reliable overall classification rate across the out-of-time population |
| **PR-AUC** | **0.2371** | Robust precision-recall curve baseline for highly imbalanced bankruptcy events |
| **Brier Score** | **0.1529** | Low probability calibration error |
| **Calibrated Threshold** | **~0.542** | Learned from Out-Of-Fold F1 maximization on training data |
| **Selected Champion Model** | **CatBoost** | Best performer identified through Optuna Bayesian optimization |

---

## 💻 Tech Stack

| Domain | Technology / Framework |
| :--- | :--- |
| **Programming Language** | Python 3.12+ |
| **Data & Feature Engineering** | Pandas, NumPy, Scikit-Learn |
| **Model Architectures** | CatBoost, XGBoost, LightGBM, Scikit-Learn Ensembles |
| **Optimization & Tuning** | Optuna (Tree-structured Parzen Estimator) |
| **Experiment Tracking** | MLflow |
| **API Framework** | FastAPI, Uvicorn, Pydantic |
| **User Interface** | Streamlit |
| **Containerization** | Docker, Docker Compose |
| **Package Management** | uv / pyproject.toml |

---
