"""
Supply Chain AI Operations Autonomous Agent.
Coordinates data ingestion, anomaly scoring, predictive risk modeling,
and multi-domain prescriptive mitigation planning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.preprocessor import SupplyChainPreprocessor
from src.predictor import SupplyChainPredictor
from src.recommendation_engine import DisruptionRecommendationEngine

class SupplyChainAgent:
    def __init__(self, models_dir: str = "models"):
        self.predictor = SupplyChainPredictor(models_dir=models_dir)
        self.recommender = DisruptionRecommendationEngine()
        self.is_initialized = False

    def initialize(self):
        """Load trained models and configuration."""
        self.predictor.load_models()
        self.is_initialized = True
        return self

    def analyze_supply_chain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        End-to-end AI operational assessment of supply chain stream:
        1. Preprocess & predict disruption risk probabilities and scores
        2. Unsupervised isolation forest anomaly tagging
        3. Domain-specific recommendation generation
        4. Portfolio-level resilience metrics computation
        """
        if not self.is_initialized:
            self.initialize()

        scored_df = self.predictor.predict(df)
        
        # Summary statistics
        total_orders = len(scored_df)
        critical_alerts = int((scored_df["predicted_risk_level"] == "CRITICAL").sum())
        high_alerts = int((scored_df["predicted_risk_level"] == "HIGH").sum())
        anomalies_detected = int(scored_df["is_anomalous_pattern"].sum())
        avg_risk_score = round(float(scored_df["pred_risk_score"].mean()), 1)
        
        # Batch recommendations for high risk items
        high_risk_subset = scored_df[scored_df["predicted_risk_level"].isin(["CRITICAL", "HIGH"])]
        recommendations = self.recommender.batch_generate(high_risk_subset)
        
        # Aggregate financial risk exposure
        total_downtime_exposure = sum(r["financial_impact"]["estimated_potential_downtime_loss_usd"] for r in recommendations)
        total_mitigation_cost = sum(r["financial_impact"]["estimated_mitigation_cost_usd"] for r in recommendations)
        
        # Regional Breakdown
        regional_risk = scored_df.groupby("supplier_region")["pred_risk_score"].mean().round(1).to_dict()
        category_risk = scored_df.groupby("product_category")["pred_risk_score"].mean().round(1).to_dict()

        return {
            "summary": {
                "total_monitored_orders": total_orders,
                "critical_disruptions": critical_alerts,
                "high_risk_orders": high_alerts,
                "anomalies_flagged": anomalies_detected,
                "fleet_average_risk_score": avg_risk_score,
                "estimated_financial_exposure_usd": round(total_downtime_exposure, 2),
                "recommended_mitigation_budget_usd": round(total_mitigation_cost, 2),
                "expected_roi_ratio": round(total_downtime_exposure / max(1.0, total_mitigation_cost), 1)
            },
            "regional_risk_index": regional_risk,
            "category_risk_index": category_risk,
            "scored_data": scored_df,
            "high_risk_recommendations": recommendations,
            "model_metadata": self.predictor.metrics
        }

    def assess_single_event(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Real-time scoring and recommendation for a single sensor/ERP event."""
        if not self.is_initialized:
            self.initialize()
            
        df = pd.DataFrame([record])
        scored = self.predictor.predict(df)
        rec = self.recommender.generate_recommendations(scored.iloc[0].to_dict())
        
        return {
            "prediction": scored.iloc[0].to_dict(),
            "mitigation_plan": rec
        }

