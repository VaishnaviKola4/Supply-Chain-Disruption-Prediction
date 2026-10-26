# API & Interface Reference Guide

This document specifies the Python programmatic API, data contracts, and interactive user interface workflows of the **NexChain AI** operations engine.

---

## 1. Python Agentic API (`src.agent.SupplyChainAgent`)

The primary orchestration class is `SupplyChainAgent`, which loads preprocessing artifacts and trained estimators to run batch evaluations or real-time event checks.

### Class Definition

```python
from src.agent import SupplyChainAgent

agent = SupplyChainAgent(models_dir="models")
agent.initialize()
```

### Methods

#### `initialize() -> SupplyChainAgent`
Loads serialized machine learning models (`preprocessor.joblib`, `classifier.joblib`, `regressor.joblib`, `anomaly_detector.joblib`).
- **Returns:** Initialized agent instance.

#### `analyze_supply_chain(df: pd.DataFrame) -> Dict[str, Any]`
Performs full portfolio intelligence over an input pandas DataFrame of orders.

- **Parameters:**
  - `df` (*pd.DataFrame*): Table containing supply chain features (e.g. stock, lead time, delays, compliance).
- **Returns:**
  ```python
  {
      "summary": {
          "total_monitored_orders": int,
          "critical_disruptions": int,
          "high_risk_orders": int,
          "anomalies_flagged": int,
          "fleet_average_risk_score": float,
          "estimated_financial_exposure_usd": float,
          "recommended_mitigation_budget_usd": float,
          "expected_roi_ratio": float
      },
      "regional_risk_index": Dict[str, float],
      "category_risk_index": Dict[str, float],
      "scored_data": pd.DataFrame,
      "high_risk_recommendations": List[Dict[str, Any]],
      "model_metadata": Dict[str, Any]
  }
  ```

#### `assess_single_event(record: Dict[str, Any]) -> Dict[str, Any]`
Real-time single-record diagnostic inference.

- **Parameters:**
  - `record` (*Dict[str, Any]*): Dictionary representing a single shipment or PO.
- **Returns:**
  ```python
  {
      "prediction": {
          "pred_disruption_flag": int,
          "pred_disruption_probability": float,
          "pred_risk_score": float,
          "predicted_risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
          "is_anomalous_pattern": bool,
          "anomaly_severity_score": float,
          ...
      },
      "mitigation_plan": {
          "order_id": str,
          "sku": str,
          "root_causes": List[str],
          "recommended_actions": [
              {
                  "domain": "Supplier Sourcing" | "Inventory Policy" | "Logistics & Freight" | "Production Scheduling",
                  "priority": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
                  "title": str,
                  "description": str,
                  "cost_impact": str
              }
          ],
          "financial_impact": {
              "estimated_potential_downtime_loss_usd": float,
              "estimated_mitigation_cost_usd": float,
              "projected_cost_avoidance_roi": float
          }
      }
  }
  ```

---

## 2. Recommendation Engine API (`src.recommendation_engine.DisruptionRecommendationEngine`)

Synthesizes actionable mitigation directives across four manufacturing dimensions.

### Usage Example

```python
from src.recommendation_engine import DisruptionRecommendationEngine

engine = DisruptionRecommendationEngine()
plan = engine.generate_recommendations({
    "sku": "SKU-PRO-01",
    "product_name": "High-Voltage Power Inverter",
    "product_category": "Powertrain",
    "current_stock_units": 45,
    "target_stock_units": 500,
    "safety_stock_units": 125,
    "transit_delay_days": 4.5,
    "quality_defect_rate_pct": 5.2,
    "pred_risk_score": 78.4,
    "predicted_risk_level": "CRITICAL",
    "is_anomalous_pattern": True
})
```

---

## 3. Web UI Navigation Guide (Streamlit)

The web dashboard is accessed via `streamlit run src/app.py` and provides 5 specialized consoles:

| Console | Primary Action | Key Output |
| :--- | :--- | :--- |
| **Fleet Operations Dashboard** | Filter global shipments by category, origin, criticality | Real-time KPI cards, Risk vs Buffer scatter plot, Priority Mitigation Queue |
| **Real-time Disruption Inspector** | Interactive form for single shipment parameter testing | Disruption risk score gauge, anomaly detection tag, prescriptive mitigation cards |
| **What-If Scenario Simulator** | Select macro stress shocks (port strikes, shortages) | Comparative before-and-after risk distributions, financial loss deltas |
| **Model Diagnostics & Governance** | Audit machine learning estimators | ROC-AUC (0.978), R² (0.910), RMSE (4.6), Feature Importance plot |
| **Custom Batch Upload** | Ingest enterprise CSV files via drag-and-drop | Instant scoring, anomaly tagging, downloadable augmented CSV report |

