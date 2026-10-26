"""
Data Preprocessing and Feature Engineering Pipeline for Supply Chain Disruption Prediction.
Handles missing values, scaling, categorical encoding, temporal features, and dataset splitting.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os

FEATURE_COLS_NUMERIC = [
    "supplier_compliance_pct",
    "target_stock_units",
    "current_stock_units",
    "safety_stock_units",
    "stockout_buffer_ratio",
    "days_of_supply",
    "inventory_turnover",
    "lead_time_days",
    "lead_time_variance_days",
    "quality_defect_rate_pct",
    "machine_downtime_hours",
    "production_utilization_pct",
    "schedule_adherence_pct",
    "geopolitical_risk_index",
    "transit_delay_days",
    "carrier_on_time_pct",
    "order_fulfillment_rate_pct",
    "month",
    "day_of_week"
]

CATEGORICAL_COLS = [
    "supplier_region",
    "criticality",
    "transport_mode",
    "weather_risk_level"
]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal, interaction, and ratio features."""
    df = df.copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek
    else:
        df["month"] = 6
        df["day_of_week"] = 2
        
    # Safeguards for missing or zero divisions
    if "target_stock_units" in df.columns and "current_stock_units" in df.columns:
        df["stock_deficit_pct"] = np.maximum(0, (df["target_stock_units"] - df["current_stock_units"]) / np.maximum(df["target_stock_units"], 1.0)) * 100
    else:
        df["stock_deficit_pct"] = 0.0

    # Composite fragility score
    transit_delay = df["transit_delay_days"] if "transit_delay_days" in df.columns else 0.0
    defect = df["quality_defect_rate_pct"] if "quality_defect_rate_pct" in df.columns else 0.0
    buffer = df["stockout_buffer_ratio"] if "stockout_buffer_ratio" in df.columns else 1.0
    
    df["logistics_vulnerability_index"] = (transit_delay * 2.0) + defect + (10.0 / (np.maximum(buffer, 0.1)))
    return df

class SupplyChainPreprocessor:
    def __init__(self):
        self.scaler = RobustScaler()
        self.categories_map = {}
        self.feature_names = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        df_feat = engineer_features(df)
        
        # Categorical one-hot mapping structure
        for cat in CATEGORICAL_COLS:
            if cat in df_feat.columns:
                self.categories_map[cat] = sorted(df_feat[cat].dropna().unique().tolist())
                
        # Determine full feature columns
        num_cols = [c for c in FEATURE_COLS_NUMERIC if c in df_feat.columns] + ["stock_deficit_pct", "logistics_vulnerability_index"]
        
        # Get one-hot column names
        dummy_cols = []
        for cat, values in self.categories_map.items():
            for val in values:
                dummy_cols.append(f"{cat}_{val}")
                
        self.feature_names = num_cols + dummy_cols
        
        # Fit scaler on numeric features
        X_num = df_feat[num_cols].fillna(df_feat[num_cols].median())
        self.scaler.fit(X_num)
        self.numeric_cols_ = num_cols
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
            
        df_feat = engineer_features(df)
        
        # Numeric scaled features
        X_num = df_feat[self.numeric_cols_].fillna(df_feat[self.numeric_cols_].median())
        scaled_num = self.scaler.transform(X_num)
        
        # Categorical one-hot features
        cat_arrays = []
        for cat in CATEGORICAL_COLS:
            cat_vals = self.categories_map.get(cat, [])
            current_vals = df_feat[cat].astype(str).values if cat in df_feat.columns else np.array(["Unknown"] * len(df_feat))
            for val in cat_vals:
                col_bool = (current_vals == str(val)).astype(float).reshape(-1, 1)
                cat_arrays.append(col_bool)
                
        if cat_arrays:
            encoded_cats = np.hstack(cat_arrays)
            X_all = np.hstack([scaled_num, encoded_cats])
        else:
            X_all = scaled_num
            
        return X_all

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def save(self, filepath: str = "models/preprocessor.joblib"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Preprocessor saved to {filepath}")

    @staticmethod
    def load(filepath: str = "models/preprocessor.joblib"):
        return joblib.load(filepath)

if __name__ == "__main__":
    df = pd.read_csv("data/supply_chain_data.csv")
    preprocessor = SupplyChainPreprocessor()
    X = preprocessor.fit_transform(df)
    print(f"Features shape: {X.shape}")
    preprocessor.save()

