"""
Predictive Disruption Engine for Supply Chain Risk Modeling.
Implements:
1. Disruption Classification (Random Forest / Gradient Boosting)
2. Continuous Risk Severity Score Regressor
3. Unsupervised Outlier & Anomaly Detection (Isolation Forest)
4. Time Series Trend Forecasting for aggregate metrics (Moving Window / Trend projection)
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, r2_score, mean_absolute_error
)
from src.preprocessor import SupplyChainPreprocessor

class SupplyChainPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.preprocessor = None
        self.classifier = None
        self.regressor = None
        self.anomaly_detector = None
        self.metrics = {}

    def train(self, data_path: str = "data/supply_chain_data.csv") -> Dict[str, Any]:
        """Train classifier, regressor, and anomaly detector."""
        df = pd.read_csv(data_path)
        
        self.preprocessor = SupplyChainPreprocessor()
        X = self.preprocessor.fit_transform(df)
        self.preprocessor.save(os.path.join(self.models_dir, "preprocessor.joblib"))
        
        y_class = df["disruption_occurred"].values
        y_reg = df["disruption_risk_score"].values
        
        # Train / Test split
        X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
        )
        
        # 1. Classification Model (Disruption event flag)
        print("Training Random Forest Classifier...")
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        self.classifier.fit(X_train, y_train_cls)
        
        y_pred_cls = self.classifier.predict(X_test)
        y_prob_cls = self.classifier.predict_proba(X_test)[:, 1]
        
        cls_metrics = {
            "accuracy": round(float(accuracy_score(y_test_cls, y_pred_cls)), 4),
            "precision": round(float(precision_score(y_test_cls, y_pred_cls)), 4),
            "recall": round(float(recall_score(y_test_cls, y_pred_cls)), 4),
            "f1_score": round(float(f1_score(y_test_cls, y_pred_cls)), 4),
            "roc_auc": round(float(roc_auc_score(y_test_cls, y_prob_cls)), 4)
        }
        
        # 2. Continuous Disruption Risk Score Regressor
        print("Training Gradient Boosting Regressor for Disruption Risk Score...")
        self.regressor = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            random_state=42
        )
        self.regressor.fit(X_train, y_train_reg)
        
        y_pred_reg = self.regressor.predict(X_test)
        reg_metrics = {
            "rmse": round(float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))), 3),
            "mae": round(float(mean_absolute_error(y_test_reg, y_pred_reg)), 3),
            "r2_score": round(float(r2_score(y_test_reg, y_pred_reg)), 4)
        }
        
        # 3. Unsupervised Outlier & Anomaly Detection (Zero-day disruption shifts)
        print("Training Isolation Forest Anomaly Detector...")
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=0.08,
            random_state=42,
            n_jobs=-1
        )
        self.anomaly_detector.fit(X_train)
        
        self.metrics = {
            "classification": cls_metrics,
            "regression": reg_metrics,
            "feature_names": self.preprocessor.feature_names
        }
        
        self.save_models()
        return self.metrics

    def save_models(self):
        os.makedirs(self.models_dir, exist_ok=True)
        joblib.dump(self.classifier, os.path.join(self.models_dir, "classifier.joblib"))
        joblib.dump(self.regressor, os.path.join(self.models_dir, "regressor.joblib"))
        joblib.dump(self.anomaly_detector, os.path.join(self.models_dir, "anomaly_detector.joblib"))
        joblib.dump(self.metrics, os.path.join(self.models_dir, "metrics.joblib"))
        print(f"All models saved to {self.models_dir}")

    def load_models(self):
        self.preprocessor = SupplyChainPreprocessor.load(os.path.join(self.models_dir, "preprocessor.joblib"))
        self.classifier = joblib.load(os.path.join(self.models_dir, "classifier.joblib"))
        self.regressor = joblib.load(os.path.join(self.models_dir, "regressor.joblib"))
        self.anomaly_detector = joblib.load(os.path.join(self.models_dir, "anomaly_detector.joblib"))
        if os.path.exists(os.path.join(self.models_dir, "metrics.joblib")):
            self.metrics = joblib.load(os.path.join(self.models_dir, "metrics.joblib"))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run predictions on new data records and augment with predictions and anomaly scores."""
        if self.preprocessor is None or self.classifier is None:
            self.load_models()
            
        X = self.preprocessor.transform(df)
        
        disruption_preds = self.classifier.predict(X)
        disruption_probs = self.classifier.predict_proba(X)[:, 1]
        risk_scores = self.regressor.predict(X)
        
        # Isolation Forest: -1 is anomaly, 1 is inlier. We map to boolean is_anomaly and anomaly score
        anomaly_flags = self.anomaly_detector.predict(X) == -1
        anomaly_scores = -self.anomaly_detector.score_samples(X) # Higher means more anomalous
        
        result_df = df.copy()
        result_df["pred_disruption_flag"] = disruption_preds
        result_df["pred_disruption_probability"] = np.round(disruption_probs * 100, 1)
        result_df["pred_risk_score"] = np.round(risk_scores, 1)
        result_df["is_anomalous_pattern"] = anomaly_flags
        result_df["anomaly_severity_score"] = np.round(anomaly_scores * 100, 2)
        
        # Compute Risk Level tier
        def categorize_risk(score):
            if score >= 70:
                return "CRITICAL"
            elif score >= 50:
                return "HIGH"
            elif score >= 30:
                return "MEDIUM"
            return "LOW"
            
        result_df["predicted_risk_level"] = result_df["pred_risk_score"].apply(categorize_risk)
        return result_df

    def get_feature_importances(self, top_n: int = 10) -> pd.DataFrame:
        if self.classifier is None:
            self.load_models()
        importances = self.classifier.feature_importances_
        feat_df = pd.DataFrame({
            "feature": self.preprocessor.feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)
        return feat_df.head(top_n)

if __name__ == "__main__":
    predictor = SupplyChainPredictor()
    metrics = predictor.train()
    print("Training Results:")
    print("Classification:", metrics["classification"])
    print("Regression:", metrics["regression"])

