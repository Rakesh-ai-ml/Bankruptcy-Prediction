import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    ExtraTreesClassifier
)
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    brier_score_loss
)
import optuna
import joblib
import mlflow
import mlflow.sklearn
from src.BancrupcyClassification.entity.config_entity import ModelTrainerConfig
from src.BancrupcyClassification.logger import logger


class ModelTrainer:

    def __init__(self, config: ModelTrainerConfig, n_trials: int = 100):
        self.config = config
        self.n_trials = n_trials
        self.rs = 42
        self.X_train = None
        self.y_train = None
        self.groups = None
        self.best_threshold = 0.5
        self.classifiers = [
            "RandomForest", "LightGBM", "CatBoost", "HistGradientBoost",
            "XGBoost", "ExtraTrees", "LogisticRegression"
        ]

    def load_train_data(self):
        train_ds = pd.read_csv(self.config.train_data_path)

        if 'company_name' in train_ds.columns:
            self.groups = train_ds['company_name']
            self.X_train = train_ds.drop([self.config.target_column, 'company_name'], axis=1)
        else:
            self.groups = None
            self.X_train = train_ds.drop([self.config.target_column], axis=1)

        self.y_train = train_ds[self.config.target_column]
        logger.info(
            f"Loaded train data: {self.X_train.shape}, Unique groups: {self.groups.nunique() if self.groups is not None else 'None'}"
        )

    def get_classifier(self, trial: optuna.trial.Trial):
        classifier_name = trial.suggest_categorical("classifier", self.classifiers)

        if classifier_name == "RandomForest":
            clf = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 50),
                class_weight="balanced",
                random_state=self.rs,
                n_jobs=-1
            )

        elif classifier_name == "LightGBM":
            clf = LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 5),
                num_leaves=trial.suggest_int("num_leaves", 10, 31),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                class_weight="balanced",
                random_state=self.rs,
                verbose=-1
            )

        elif classifier_name == "CatBoost":
            clf = CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 100, 300, step=50),
                depth=trial.suggest_int("depth", 3, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                auto_class_weights="Balanced",
                random_state=self.rs,
                verbose=0
            )

        elif classifier_name == "HistGradientBoost":
            clf = HistGradientBoostingClassifier(
                max_iter=trial.suggest_int("max_iter", 100, 300, step=50),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 50),
                class_weight="balanced",
                random_state=self.rs
            )

        elif classifier_name == "XGBoost":
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos = neg_count / max(pos_count, 1)

            clf = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 2, 5),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                scale_pos_weight=scale_pos,
                random_state=self.rs,
                eval_metric="logloss"
            )

        elif classifier_name == "ExtraTrees":
            clf = ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300, step=50),
                max_depth=trial.suggest_int("max_depth", 3, 6),
                min_samples_split=trial.suggest_int("min_samples_split", 5, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 50),
                class_weight="balanced",
                random_state=self.rs,
                n_jobs=-1
            )

        elif classifier_name == "LogisticRegression":
            clf = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 10.0, log=True),
                solver="lbfgs",
                max_iter=500,
                class_weight="balanced",
                random_state=self.rs
            )
        return classifier_name, clf

    def objective(self, trial: optuna.trial.Trial):
        _, clf = self.get_classifier(trial)

        if self.groups is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(
                clf, self.X_train, self.y_train,
                groups=self.groups, cv=cv, scoring="f1", n_jobs=-1
            )
        else:
            scores = cross_val_score(
                clf, self.X_train, self.y_train,
                cv=5, scoring="roc_auc", n_jobs=-1
            )
        return scores.mean()

    def build_classifier_instance(self, model_name: str, params: dict):
        if model_name == "RandomForest":
            return RandomForestClassifier(**params, class_weight="balanced", random_state=self.rs, n_jobs=-1)
        elif model_name == "LightGBM":
            return LGBMClassifier(**params, class_weight="balanced", random_state=self.rs, verbose=-1)
        elif model_name == "CatBoost":
            return CatBoostClassifier(**params, auto_class_weights="Balanced", random_state=self.rs, verbose=0)
        elif model_name == "HistGradientBoost":
            return HistGradientBoostingClassifier(**params, class_weight="balanced", random_state=self.rs)
        elif model_name == "XGBoost":
            neg_count = (self.y_train == 0).sum()
            pos_count = (self.y_train == 1).sum()
            scale_pos = neg_count / max(pos_count, 1)
            return XGBClassifier(**params, scale_pos_weight=scale_pos, random_state=self.rs, eval_metric="logloss")
        elif model_name == "ExtraTrees":
            return ExtraTreesClassifier(**params, class_weight="balanced", random_state=self.rs, n_jobs=-1)
        elif model_name == "LogisticRegression":
            return LogisticRegression(**params, class_weight="balanced", random_state=self.rs, max_iter=500)
        else:
            raise ValueError(f"Unknown classifier: {model_name}")

    def calibrate_threshold(self, model):
        
        if self.groups is None:                                                                                                                                                  
                return 0.5                                                                                                                                                           
                                                                                                                                                                                     
        gkf = GroupKFold(n_splits=5)                                                                                                                                             
        oof_probas = cross_val_predict(                                                                                                                                          
            model, self.X_train, self.y_train,                                                                                                                                   
            groups=self.groups, cv=gkf, method="predict_proba", n_jobs=-1                                                                                                        
        )[:, 1]                                                                                                                                                                  
                                                                                                                                                                                    
        precisions, recalls, thresholds = precision_recall_curve(self.y_train, oof_probas)                                                                                       
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)                                                                                                   
        best_idx = np.argmax(f1_scores)                                                                                                                                          
        best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])                                                                                                   
                                                                                                                                                                                    
        logger.info(                                                                                                                                                             
            f"Calibrated best threshold from GroupKFold OOF: {best_threshold:.4f} (CV F1: {f1_scores[best_idx]:.4f})"                                                            
        )                                                                                                                                                                        
        return best_threshold    
    def evaluate(self, model, threshold: float = 0.5):
        y_proba = model.predict_proba(self.X_train)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(self.y_train, y_pred)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0.0

        results = {
            "Accuracy": accuracy_score(self.y_train, y_pred),
            "Precision": precision_score(self.y_train, y_pred, zero_division=0),
            "Recall": recall_score(self.y_train, y_pred, zero_division=0),
            "Specificity": specificity,
            "F1-Score": f1_score(self.y_train, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(self.y_train, y_proba),
            "Brier Score": brier_score_loss(self.y_train, y_proba),
        }
        return results

    def run(self):
        self.load_train_data()

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.rs))
        study.optimize(self.objective, n_trials=self.n_trials, show_progress_bar=True)

        best_model_name = study.best_trial.params["classifier"]
        best_params = {k: v for k, v in study.best_trial.params.items() if k != "classifier"}
        best_cv_auc = study.best_value

        logger.info(f"Best Model: {best_model_name} | Best CV ROC-AUC: {best_cv_auc:.4f}")
        logger.info(f"Best Params: {best_params}")

        # Instantiate and fit best model
        best_model = self.build_classifier_instance(best_model_name, best_params)
        self.best_threshold = self.calibrate_threshold(best_model)

        best_model.fit(self.X_train, self.y_train)
        results = self.evaluate(best_model, threshold=self.best_threshold)

        # Save model and threshold
        os.makedirs(self.config.root_dir, exist_ok=True)
        model_save_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(best_model, model_save_path)

        threshold_path = os.path.join(self.config.root_dir, "threshold.json")
        with open(threshold_path, "w") as f:
            json.dump({"threshold": self.best_threshold, "best_model": best_model_name}, f, indent=4)

        logger.info(f"Model saved to {model_save_path}, threshold saved to {threshold_path}")

        #logging into MLflow 
        try:
            mlflow.end_run() 
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            # mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("Bankruptcy_Classification")
            with mlflow.start_run(run_name=f"Run_{best_model_name}") as run:
                mlflow.log_metrics({k: v for k, v in results.items() if isinstance(v, (int, float))})
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_params(best_params)
                mlflow.log_metric("cv_roc_auc", best_cv_auc)
                mlflow.log_metric("calibrated_threshold", self.best_threshold)

                os.makedirs("artifacts", exist_ok=True)
                with open("artifacts/run_id.txt", "w") as f:
                    f.write(run.info.run_id)
        except Exception as e:
            logger.warning(f"MLflow tracking skipped (local server at http://127.0.0.1:5000 not reachable): {e}")
