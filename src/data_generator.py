"""
Data Generator Module for Manufacturing Supply Chain Disruption Prediction.
Generates realistic multi-tier supply chain data encompassing:
1. Supplier performance (lead time variance, quality defect rate, compliance score)
2. Inventory levels (safety stock buffer, warehouse capacity utilization, days of supply)
3. Production metrics (machine downtime, schedule adherence, capacity utilization)
4. Delivery/logistics (transit delays, carrier on-time rate, weather/geopolitical risk index)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_supply_chain_data(
    n_samples: int = 3500,
    random_seed: int = 42,
    save_path: str = "data/supply_chain_data.csv"
) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    start_date = datetime(2025, 1, 1)
    date_offsets = np.random.randint(0, 365, size=n_samples)
    dates = [start_date + timedelta(days=int(offset)) for offset in date_offsets]
    
    suppliers = [
        {"id": "SUP-101", "name": "Apex Microtronics", "tier": 1, "region": "East Asia", "base_lead_time": 18, "reliability": 0.94},
        {"id": "SUP-102", "name": "Vanguard Precision Alloys", "tier": 1, "region": "North America", "base_lead_time": 8, "reliability": 0.97},
        {"id": "SUP-103", "name": "Bavaria AutoComponents", "tier": 1, "region": "Europe", "base_lead_time": 14, "reliability": 0.92},
        {"id": "SUP-104", "name": "Pacific Polymer Dynamics", "tier": 2, "region": "Southeast Asia", "base_lead_time": 25, "reliability": 0.86},
        {"id": "SUP-105", "name": "SinoTech Silicon Systems", "tier": 1, "region": "East Asia", "base_lead_time": 22, "reliability": 0.88},
        {"id": "SUP-106", "name": "Nordic Sensor Corp", "tier": 2, "region": "Europe", "base_lead_time": 12, "reliability": 0.95},
        {"id": "SUP-107", "name": "Iberia Logistics & Foundry", "tier": 2, "region": "Europe", "base_lead_time": 16, "reliability": 0.89},
        {"id": "SUP-108", "name": "Atlas Raw Materials", "tier": 3, "region": "South America", "base_lead_time": 30, "reliability": 0.82},
    ]
    
    products = [
        {"sku": "SKU-PRO-01", "name": "High-Voltage Power Inverter", "category": "Powertrain", "target_stock": 500, "criticality": "Critical"},
        {"sku": "SKU-PRO-02", "name": "Autonomous Sensor Array", "category": "Electronics", "target_stock": 750, "criticality": "High"},
        {"sku": "SKU-PRO-03", "name": "Structural Composite Chassis", "category": "Body", "target_stock": 300, "criticality": "Critical"},
        {"sku": "SKU-PRO-04", "name": "Industrial Microcontroller Unit", "category": "Electronics", "target_stock": 1200, "criticality": "High"},
        {"sku": "SKU-PRO-05", "name": "Precision Hydraulic Valve Set", "category": "Hydraulics", "target_stock": 450, "criticality": "Medium"},
        {"sku": "SKU-PRO-06", "name": "Thermal Dissipation Radiator", "category": "Cooling", "target_stock": 600, "criticality": "Medium"},
    ]
    
    transport_modes = ["Ocean Freight", "Air Cargo", "Intermodal Rail", "Express Road Freight"]
    weather_risk_levels = ["Low", "Moderate", "Severe"]
    
    records = []
    
    for i in range(n_samples):
        record_date = dates[i]
        supplier = np.random.choice(suppliers)
        product = np.random.choice(products)
        mode = np.random.choice(transport_modes, p=[0.35, 0.20, 0.25, 0.20])
        weather_risk = np.random.choice(weather_risk_levels, p=[0.65, 0.25, 0.10])
        
        # Lead Time & Variance
        base_lead = supplier["base_lead_time"]
        mode_multiplier = 0.4 if mode == "Air Cargo" else (1.0 if mode == "Express Road Freight" else (1.2 if mode == "Intermodal Rail" else 1.8))
        lead_time = max(2.0, np.random.normal(base_lead * mode_multiplier, 3.5))
        lead_time_variance = round(lead_time - (base_lead * mode_multiplier), 2)
        
        # Quality & Supplier Metrics
        defect_rate = np.random.beta(2, 45) if supplier["reliability"] > 0.90 else np.random.beta(4, 30)
        defect_rate = round(float(defect_rate) * 100, 2) # percentage
        supplier_historical_compliance = round(supplier["reliability"] * 100 + np.random.normal(0, 2.5), 1)
        supplier_historical_compliance = min(99.9, max(60.0, supplier_historical_compliance))
        
        # Inventory Metrics
        target_stock = product["target_stock"]
        current_stock = int(np.random.normal(target_stock * 0.85, target_stock * 0.25))
        current_stock = max(10, current_stock)
        safety_stock = int(target_stock * 0.25)
        stockout_buffer_ratio = round(current_stock / max(1, safety_stock), 2)
        daily_consumption = np.random.uniform(15, 45)
        days_of_supply = round(current_stock / daily_consumption, 1)
        inventory_turnover_rate = round(np.random.uniform(4.0, 14.0), 2)
        
        # Production / Manufacturing Metrics
        machine_downtime_hours = max(0.0, np.random.exponential(4.0) if np.random.rand() > 0.4 else 0.0)
        machine_downtime_hours = round(machine_downtime_hours, 1)
        production_utilization = round(min(98.0, max(50.0, np.random.normal(82, 9))), 1)
        schedule_adherence_rate = round(min(100.0, max(60.0, np.random.normal(91, 7))), 1)
        
        # Delivery & Logistics Metrics
        base_transit_delay = max(0.0, np.random.exponential(1.5))
        if weather_risk == "Severe":
            base_transit_delay += np.random.uniform(3.0, 8.0)
        elif weather_risk == "Moderate":
            base_transit_delay += np.random.uniform(1.0, 3.0)
        if mode == "Ocean Freight" and np.random.rand() < 0.2:
            base_transit_delay += np.random.uniform(4.0, 10.0) # port congestion
            
        transit_delay_days = round(base_transit_delay, 1)
        carrier_on_time_performance = round(min(99.0, max(60.0, 95.0 - (transit_delay_days * 3.2) + np.random.normal(0, 3))), 1)
        order_fulfillment_rate = round(min(100.0, max(55.0, 100.0 - (defect_rate * 1.5) - (transit_delay_days * 2.0) + np.random.normal(0, 2))), 1)
        
        # Geopolitical / Port Congestion Index (1 to 10)
        geo_risk_score = 7.5 if supplier["region"] == "East Asia" and np.random.rand() < 0.25 else np.random.uniform(1.0, 5.0)
        geo_risk_score = round(geo_risk_score, 1)
        
        # Calculate Ground Truth Disruption Risk Score & Binary Event Flag
        # Disruption triggers when inventory is low, lead time variance is high, defect rate is elevated, or transit delay is severe
        risk_components = [
            (1.0 if stockout_buffer_ratio < 1.0 else 0.0) * 35,
            (min(1.0, max(0.0, (lead_time_variance + 3) / 10))) * 25,
            (min(1.0, defect_rate / 8.0)) * 15,
            (min(1.0, transit_delay_days / 6.0)) * 25,
            (1.0 if weather_risk == "Severe" else (0.5 if weather_risk == "Moderate" else 0.0)) * 10,
            (machine_downtime_hours / 24.0) * 15
        ]
        
        risk_score = min(100.0, max(5.0, sum(risk_components) + np.random.normal(0, 4.0)))
        risk_score = round(risk_score, 1)
        
        # Disruption occurs if risk score >= 55 or severe stockout with delay
        disrupted = 1 if (risk_score >= 55.0 or (stockout_buffer_ratio < 0.8 and transit_delay_days >= 3.0)) else 0
        
        severity_category = "Low"
        if risk_score >= 70:
            severity_category = "Critical"
        elif risk_score >= 50:
            severity_category = "High"
        elif risk_score >= 30:
            severity_category = "Medium"
            
        records.append({
            "order_id": f"ORD-2025-{10000 + i}",
            "date": record_date.strftime("%Y-%m-%d"),
            "supplier_id": supplier["id"],
            "supplier_name": supplier["name"],
            "supplier_tier": supplier["tier"],
            "supplier_region": supplier["region"],
            "supplier_compliance_pct": supplier_historical_compliance,
            "sku": product["sku"],
            "product_name": product["name"],
            "product_category": product["category"],
            "criticality": product["criticality"],
            "target_stock_units": target_stock,
            "current_stock_units": current_stock,
            "safety_stock_units": safety_stock,
            "stockout_buffer_ratio": stockout_buffer_ratio,
            "days_of_supply": days_of_supply,
            "inventory_turnover": inventory_turnover_rate,
            "transport_mode": mode,
            "lead_time_days": round(lead_time, 1),
            "lead_time_variance_days": lead_time_variance,
            "quality_defect_rate_pct": defect_rate,
            "machine_downtime_hours": machine_downtime_hours,
            "production_utilization_pct": production_utilization,
            "schedule_adherence_pct": schedule_adherence_rate,
            "weather_risk_level": weather_risk,
            "geopolitical_risk_index": geo_risk_score,
            "transit_delay_days": transit_delay_days,
            "carrier_on_time_pct": carrier_on_time_performance,
            "order_fulfillment_rate_pct": order_fulfillment_rate,
            "disruption_risk_score": risk_score,
            "severity_category": severity_category,
            "disruption_occurred": disrupted
        })
        
    df = pd.DataFrame(records)
    df.sort_values(by="date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Generated {len(df)} supply chain records saved to {save_path}")
        
    return df

if __name__ == "__main__":
    generate_supply_chain_data()

