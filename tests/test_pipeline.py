import pytest
import pandas as pd
import numpy as np
import os
from src.data_generator import generate_supply_chain_data
from src.preprocessor import SupplyChainPreprocessor
from src.predictor import SupplyChainPredictor
from src.recommendation_engine import DisruptionRecommendationEngine
from src.agent import SupplyChainAgent

def test_data_generation():
    df = generate_supply_chain_data(n_samples=50, random_seed=99, save_path=None)
    assert len(df) == 50
    assert "disruption_occurred" in df.columns
    assert "disruption_risk_score" in df.columns
    assert "stockout_buffer_ratio" in df.columns
    assert "lead_time_days" in df.columns

def test_preprocessor():
    df = generate_supply_chain_data(n_samples=60, random_seed=42, save_path=None)
    prep = SupplyChainPreprocessor()
    X = prep.fit_transform(df)
    assert X.shape[0] == 60
    assert X.shape[1] > 20
    assert not np.isnan(X).any()

def test_predictor_train_predict(tmp_path):
    temp_dir = str(tmp_path / "models")
    os.makedirs(temp_dir, exist_ok=True)
    df = generate_supply_chain_data(n_samples=150, random_seed=12, save_path=None)
    temp_csv = str(tmp_path / "data.csv")
    df.to_csv(temp_csv, index=False)
    
    predictor = SupplyChainPredictor(models_dir=temp_dir)
    metrics = predictor.train(data_path=temp_csv)
    
    assert "classification" in metrics
    assert "regression" in metrics
    assert metrics["classification"]["roc_auc"] > 0.8
    
    test_df = df.head(10).copy()
    scored = predictor.predict(test_df)
    assert "pred_disruption_flag" in scored.columns
    assert "pred_disruption_probability" in scored.columns
    assert "pred_risk_score" in scored.columns
    assert "is_anomalous_pattern" in scored.columns

def test_recommendation_engine():
    engine = DisruptionRecommendationEngine()
    record = {
        "order_id": "ORD-TEST-999",
        "sku": "SKU-PRO-01",
        "product_name": "High-Voltage Power Inverter",
        "product_category": "Powertrain",
        "supplier_name": "Apex Microtronics",
        "current_stock_units": 40,
        "target_stock_units": 500,
        "safety_stock_units": 125,
        "stockout_buffer_ratio": 0.32,
        "transit_delay_days": 5.0,
        "quality_defect_rate_pct": 6.0,
        "transport_mode": "Ocean Freight",
        "pred_risk_score": 82.0,
        "pred_disruption_probability": 94.0,
        "predicted_risk_level": "CRITICAL",
        "is_anomalous_pattern": True
    }
    rec = engine.generate_recommendations(record)
    assert rec["risk_level"] == "CRITICAL"
    assert len(rec["recommended_actions"]) >= 3
    assert any("Air Cargo" in a["description"] or "Air Cargo" in a["title"] for a in rec["recommended_actions"])
    assert rec["financial_impact"]["estimated_potential_downtime_loss_usd"] > 0

def test_supply_chain_agent():
    agent = SupplyChainAgent(models_dir="models")
    agent.initialize()
    df = pd.read_csv("data/supply_chain_data.csv").head(15)
    res = agent.analyze_supply_chain(df)
    assert res["summary"]["total_monitored_orders"] == 15
    assert "regional_risk_index" in res
    assert "category_risk_index" in res

