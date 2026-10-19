import os
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    roc_curve,
    precision_recall_curve
)
import joblib
import mlflow
import mlflow.sklearn
from src.BancrupcyClassification.entity.config_entity import ModelEvaluationConfig
from src.BancrupcyClassification.logger import logger


class ModelEvaluation:
    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_proba = None
        self.features = None
        self.threshold = 0.5

    def load_calibrated_threshold(self) -> float:

        model_dir = os.path.dirname(self.config.model_path)
        threshold_file = os.path.join(model_dir, "threshold.json")
        if os.path.exists(threshold_file):
            try:
                with open(threshold_file, "r") as f:
                    data = json.load(f)
                    threshold = float(data.get("threshold", 0.5))
                    logger.info(f"Loaded calibrated threshold from {threshold_file}: {threshold:.4f}")
                    return threshold
            except Exception as e:
                logger.warning(f"Could not read threshold.json, defaulting to 0.5: {e}")
        return 0.5

    def evaluate(self) -> Dict[str, Any]:
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        cols_to_drop = [self.config.target_column]
        if 'company_name' in test_data.columns:
            cols_to_drop.append('company_name')

        X_test = test_data.drop(cols_to_drop, axis=1)
        y_test = test_data[self.config.target_column]

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-9)
        else:
            y_proba = model.predict(X_test).astype(float)

        # Load calibrated threshold
        self.threshold = self.load_calibrated_threshold()
        y_pred = (y_proba >= self.threshold).astype(int)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            specificity = None

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        brier = brier_score_loss(y_test, y_proba)

        results = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "Specificity": float(specificity) if specificity is not None else None,
            "F1-Score": float(f1),
            "ROC-AUC": float(roc_auc),
            "PR-AUC": float(pr_auc),
            "Brier Score": float(brier),
            "Threshold": float(self.threshold)
        }

        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.features = list(X_test.columns)

        # Save metrics 
        os.makedirs(self.config.root_dir, exist_ok=True)
        with open(self.config.metric_file_name, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Evaluation results saved to {self.config.metric_file_name}: {results}")
        return results

    def log_plots_to_mlflow(self, run=None):
        os.makedirs(self.config.root_dir, exist_ok=True)

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(self.y_test, self.y_proba):.3f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        roc_path = os.path.join(self.config.root_dir, "roc_curve.png")
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f"PR (AUC = {average_precision_score(self.y_test, self.y_proba):.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        pr_path = os.path.join(self.config.root_dir, "precision_recall_curve.png")
        plt.savefig(pr_path, bbox_inches='tight')
        plt.close()

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Pred Alive", "Pred Failed"],
                    yticklabels=["True Alive", "True Failed"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix (Thresh = {self.threshold:.2f})")
        cm_path = os.path.join(self.config.root_dir, "confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        # Feature Importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]  # Top 15 features
            plt.figure(figsize=(8, 5))
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), [self.features[i] for i in indices], rotation=75, ha='right')
            plt.title("Top Feature Importances")
            plt.tight_layout()
            fi_path = os.path.join(self.config.root_dir, "feature_importance.png")
            plt.savefig(fi_path, bbox_inches='tight')
            plt.close()

        # Log to MLflow 
        if run is not None:
            try:
                mlflow.log_artifact(roc_path)
                mlflow.log_artifact(pr_path)
                mlflow.log_artifact(cm_path)
                if os.path.exists(os.path.join(self.config.root_dir, "feature_importance.png")):
                    mlflow.log_artifact(os.path.join(self.config.root_dir, "feature_importance.png"))
            except Exception as e:
                logger.warning(f"Failed to log artifacts to MLflow: {e}")

    def log_into_mlflow(self):
        results = self.evaluate()
        self.log_plots_to_mlflow(run=None)

        # log to  MLflow 
        try:
            mlflow.end_run()
            mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
            
            active_run = mlflow.start_run(run_name="Eval_run")
            with active_run as run:
                metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float))}
                mlflow.log_metrics(metrics_to_log)
                self.log_plots_to_mlflow(run=run)
                logger.info(f"Successfully logged evaluation metrics to MLflow run {run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow tracking skipped (local server at http://127.0.0.1:5000 not reachable): {e}")

