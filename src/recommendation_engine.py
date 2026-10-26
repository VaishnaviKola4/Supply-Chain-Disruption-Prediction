"""
Recommendation Engine for Supply Chain Disruption Mitigation.
Generates multi-faceted, prescriptive mitigation actions across:
1. Alternate Supplier Allocation (Dynamic dual-sourcing)
2. Inventory Adjustments & Safety Buffer Recalibration
3. Transportation & Expedited Logistics Rerouting
4. Production Schedule Throttling & Line Rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Knowledge Base of Alternative Suppliers & Fallback Capabilities
SUPPLIER_CATALOG = {
    "Powertrain": [
        {"name": "Vanguard Precision Alloys", "region": "North America", "lead_time_days": 8, "cost_premium_pct": 12, "reliability": 0.97},
        {"name": "Bavaria AutoComponents", "region": "Europe", "lead_time_days": 14, "cost_premium_pct": 8, "reliability": 0.92},
        {"name": "Apex Microtronics", "region": "East Asia", "lead_time_days": 18, "cost_premium_pct": 0, "reliability": 0.94}
    ],
    "Electronics": [
        {"name": "Nordic Sensor Corp", "region": "Europe", "lead_time_days": 12, "cost_premium_pct": 10, "reliability": 0.95},
        {"name": "SinoTech Silicon Systems", "region": "East Asia", "lead_time_days": 22, "cost_premium_pct": -5, "reliability": 0.88},
        {"name": "Apex Microtronics", "region": "East Asia", "lead_time_days": 18, "cost_premium_pct": 0, "reliability": 0.94}
    ],
    "Body": [
        {"name": "Vanguard Precision Alloys", "region": "North America", "lead_time_days": 8, "cost_premium_pct": 15, "reliability": 0.97},
        {"name": "Atlas Raw Materials", "region": "South America", "lead_time_days": 30, "cost_premium_pct": -10, "reliability": 0.82}
    ],
    "Hydraulics": [
        {"name": "Bavaria AutoComponents", "region": "Europe", "lead_time_days": 14, "cost_premium_pct": 6, "reliability": 0.92},
        {"name": "Iberia Logistics & Foundry", "region": "Europe", "lead_time_days": 16, "cost_premium_pct": 2, "reliability": 0.89}
    ],
    "Cooling": [
        {"name": "Pacific Polymer Dynamics", "region": "Southeast Asia", "lead_time_days": 25, "cost_premium_pct": -8, "reliability": 0.86},
        {"name": "Iberia Logistics & Foundry", "region": "Europe", "lead_time_days": 16, "cost_premium_pct": 4, "reliability": 0.89}
    ]
}

class DisruptionRecommendationEngine:
    def __init__(self):
        pass

    def generate_recommendations(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single disruption prediction record and synthesize concrete prescriptive actions.
        """
        risk_score = float(record.get("pred_risk_score", record.get("disruption_risk_score", 0.0)))
        risk_level = record.get("predicted_risk_level", record.get("severity_category", "LOW")).upper()
        disruption_prob = float(record.get("pred_disruption_probability", 0.0))
        is_anomaly = bool(record.get("is_anomalous_pattern", False))
        
        sku = record.get("sku", "Unknown SKU")
        product_name = record.get("product_name", "Product")
        product_cat = record.get("product_category", "Electronics")
        current_supplier = record.get("supplier_name", "Current Supplier")
        current_stock = float(record.get("current_stock_units", 100))
        target_stock = float(record.get("target_stock_units", 200))
        safety_stock = float(record.get("safety_stock_units", 50))
        stock_buffer = float(record.get("stockout_buffer_ratio", current_stock / max(1.0, safety_stock)))
        lead_time = float(record.get("lead_time_days", 14))
        transit_delay = float(record.get("transit_delay_days", 0.0))
        defect_rate = float(record.get("quality_defect_rate_pct", 1.0))
        transport_mode = record.get("transport_mode", "Ocean Freight")
        
        actions: List[Dict[str, Any]] = []
        financial_impact = {
            "estimated_stockout_loss_usd": 0.0,
            "estimated_mitigation_cost_usd": 0.0,
            "roi_ratio": 1.0
        }
        
        # Primary Root Causes Detection
        root_causes = []
        if stock_buffer < 1.0:
            root_causes.append(f"Depleted buffer stock ({stock_buffer:.2f}x below threshold)")
        if transit_delay >= 3.0:
            root_causes.append(f"Severe transit delay ({transit_delay:.1f} days)")
        if defect_rate >= 5.0:
            root_causes.append(f"Elevated incoming defect rate ({defect_rate:.1f}%)")
        if is_anomaly:
            root_causes.append("Unusual sensor or delivery outlier signature detected by Isolation Forest")
        if not root_causes and risk_score >= 40:
            root_causes.append("Cumulative lead-time variance and regional geopolitical headwinds")
            
        # 1. Alternative Supplier Recommendation (Dual-sourcing)
        cat_suppliers = SUPPLIER_CATALOG.get(product_cat, SUPPLIER_CATALOG["Electronics"])
        alt_suppliers = [s for s in cat_suppliers if s["name"] != current_supplier]
        
        if risk_score >= 45 or defect_rate >= 4.5 or transit_delay >= 4.0:
            if alt_suppliers:
                best_alt = sorted(alt_suppliers, key=lambda x: (x["lead_time_days"], -x["reliability"]))[0]
                actions.append({
                    "domain": "Supplier Sourcing",
                    "priority": "HIGH" if risk_score >= 65 else "MEDIUM",
                    "title": f"Activate Dual-Sourcing with {best_alt['name']}",
                    "description": (
                        f"Reroute 35% of upcoming PO requirements to {best_alt['name']} ({best_alt['region']}). "
                        f"Expected lead time: {best_alt['lead_time_days']} days (vs current {lead_time:.0f} days) "
                        f"with historical reliability of {int(best_alt['reliability']*100)}%."
                    ),
                    "cost_impact": f"+{best_alt['cost_premium_pct']}% unit procurement variance"
                })
                
        # 2. Inventory Adjustment & Safety Buffer Recalibration
        if stock_buffer < 1.2 or risk_score >= 50:
            recommended_buffer_boost = int(target_stock * 0.25)
            new_safety_stock = int(safety_stock + recommended_buffer_boost)
            urgency = "CRITICAL" if stock_buffer < 0.7 else "HIGH"
            actions.append({
                "domain": "Inventory Policy",
                "priority": urgency,
                "title": f"Elevate Safety Stock Buffer by {recommended_buffer_boost} units",
                "description": (
                    f"Recalibrate safety stock parameter from {int(safety_stock)} to {new_safety_stock} units. "
                    f"Immediate emergency spot PO for {int(target_stock - current_stock + recommended_buffer_boost)} units recommended "
                    f"to avert assembly line starvation."
                ),
                "cost_impact": f"Working capital lockup: ~${recommended_buffer_boost * 120:,}"
            })

        # 3. Transportation Expedited Logistics Solutions
        if transit_delay >= 2.0 or (transport_mode == "Ocean Freight" and risk_score >= 55):
            expedited_mode = "Air Cargo" if transport_mode in ["Ocean Freight", "Intermodal Rail"] else "Dedicated Hot-Shot Express"
            actions.append({
                "domain": "Logistics & Freight",
                "priority": "CRITICAL" if risk_score >= 70 else "HIGH",
                "title": f"Convert Inbound Pipeline to {expedited_mode}",
                "description": (
                    f"Switch critical inbound batch from {transport_mode} to {expedited_mode}. "
                    f"Estimated transit compression: 6-12 days savings, mitigating line shutdown."
                ),
                "cost_impact": "Logistics expediting surcharge: ~$4,500 - $8,200"
            })
            
        # 4. Production Balancing & Schedule Throttling
        if risk_score >= 65:
            actions.append({
                "domain": "Production Scheduling",
                "priority": "MEDIUM",
                "title": "Rebalance Shift Output & Secondary Assembly Routing",
                "description": (
                    f"Slow line throughput on {product_name} by 15% across shifts A/B for next 72 hours; "
                    "reallocate active robotic cells to alternative SKUs unaffected by supplier bottlenecks."
                ),
                "cost_impact": "Operational downtime prevention savings: ~$35,000"
            })

        # If no severe actions required
        if not actions:
            actions.append({
                "domain": "Routine Monitoring",
                "priority": "LOW",
                "title": "Maintain Baseline Parameter Calibration",
                "description": "All parameters remain within nominal tolerance corridors. Maintain standard automated daily sensor polling.",
                "cost_impact": "No incremental variance"
            })

        # Estimate potential financial impact
        estimated_loss = 0.0
        if risk_score >= 70:
            estimated_loss = np.round(float(current_stock * 250 + 45000), 2)
        elif risk_score >= 50:
            estimated_loss = np.round(float(current_stock * 110 + 15000), 2)
        elif risk_score >= 30:
            estimated_loss = np.round(float(current_stock * 35), 2)
            
        mitigation_cost = np.round(estimated_loss * 0.18, 2)
        roi = round(estimated_loss / max(1.0, mitigation_cost), 1)

        return {
            "order_id": record.get("order_id", "N/A"),
            "sku": sku,
            "product_name": product_name,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "disruption_probability_pct": disruption_prob,
            "is_anomaly": is_anomaly,
            "root_causes": root_causes,
            "recommended_actions": actions,
            "financial_impact": {
                "estimated_potential_downtime_loss_usd": estimated_loss,
                "estimated_mitigation_cost_usd": mitigation_cost,
                "projected_cost_avoidance_roi": roi
            }
        }

    def batch_generate(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for _, row in df.iterrows():
            results.append(self.generate_recommendations(row.to_dict()))
        return results

if __name__ == "__main__":
    test_record = {
        "order_id": "ORD-TEST-001",
        "sku": "SKU-PRO-01",
        "product_name": "High-Voltage Power Inverter",
        "product_category": "Powertrain",
        "supplier_name": "Apex Microtronics",
        "current_stock_units": 85,
        "target_stock_units": 500,
        "safety_stock_units": 125,
        "stockout_buffer_ratio": 0.68,
        "transit_delay_days": 4.5,
        "quality_defect_rate_pct": 5.2,
        "transport_mode": "Ocean Freight",
        "pred_risk_score": 78.4,
        "pred_disruption_probability": 89.2,
        "predicted_risk_level": "CRITICAL",
        "is_anomalous_pattern": True
    }
    engine = DisruptionRecommendationEngine()
    recs = engine.generate_recommendations(test_record)
    import pprint
    pprint.pprint(recs)
