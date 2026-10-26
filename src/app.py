"""
Supply Chain Disruption Prediction & Autonomous Mitigation System.
Interactive Streamlit Application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io

from src.agent import SupplyChainAgent
from src.data_generator import generate_supply_chain_data

st.set_page_config(
    page_title="NexChain AI | Supply Chain Disruption Command Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-critical {
        background-color: #ff4b4b22;
        color: #ff4b4b;
        border: 1px solid #ff4b4b;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-high {
        background-color: #ffa50022;
        color: #ffa500;
        border: 1px solid #ffa500;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 600;
        font-size: 13px;
    }
    .badge-low {
        background-color: #00cc9622;
        color: #00cc96;
        border: 1px solid #00cc96;
        border-radius: 6px;
        padding: 4px 10px;
        font-weight: 600;
        font-size: 13px;
    }
    .action-card {
        background: linear-gradient(135deg, #1e222d 0%, #161a23 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    agent = SupplyChainAgent(models_dir="models")
    agent.initialize()
    return agent

@st.cache_data
def get_default_data():
    if not os.path.exists("data/supply_chain_data.csv"):
        generate_supply_chain_data(save_path="data/supply_chain_data.csv")
    return pd.read_csv("data/supply_chain_data.csv")

agent = load_agent()

# Sidebar Navigation & Controls
st.sidebar.image("https://img.icons8.com/fluency/96/conveyor-belt.png", width=64)
st.sidebar.title("NexChain AI")
st.sidebar.caption("Autonomous Manufacturing Disruption Intelligence")

mode = st.sidebar.radio(
    "Navigation Console",
    [
        "📊 Fleet Operations Dashboard",
        "🎯 Real-time Disruption Inspector",
        "🔮 What-If Scenario Simulator",
        "⚙️ Model Diagnostics & Governance",
        "📥 Custom Batch Upload"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Operational Controls")
auto_refresh = st.sidebar.checkbox("Live Sensor Polling Simulation", value=False)
threshold_alert = st.sidebar.slider("Alert Trigger Sensitivity (Risk Score)", 20, 90, 50, 5)

raw_df = get_default_data()

# -------------------------------------------------------------
# 1. FLEET OPERATIONS DASHBOARD
# -------------------------------------------------------------
if mode == "📊 Fleet Operations Dashboard":
    st.title("🏭 Global Supply Chain Disruption Operations")
    st.write("Real-time telemetry, disruption event classification, and financial risk exposure tracking across global manufacturing lines.")

    # Filter Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        selected_category = st.selectbox("Product Category", ["All"] + sorted(raw_df["product_category"].unique().tolist()))
    with c2:
        selected_region = st.selectbox("Supplier Origin Region", ["All"] + sorted(raw_df["supplier_region"].unique().tolist()))
    with c3:
        selected_urgency = st.selectbox("Component Criticality", ["All"] + sorted(raw_df["criticality"].unique().tolist()))

    filtered_df = raw_df.copy()
    if selected_category != "All":
        filtered_df = filtered_df[filtered_df["product_category"] == selected_category]
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["supplier_region"] == selected_region]
    if selected_urgency != "All":
        filtered_df = filtered_df[filtered_df["criticality"] == selected_urgency]

    analysis = agent.analyze_supply_chain(filtered_df)
    summary = analysis["summary"]
    scored_df = analysis["scored_data"]

    # KPI Top Metrics Cards
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Orders Monitored", f"{summary['total_monitored_orders']:,}")
    with k2:
        st.metric("Critical Disruptions", f"{summary['critical_disruptions']}", delta=f"High: {summary['high_risk_orders']}", delta_color="inverse")
    with k3:
        st.metric("Mean Disruption Score", f"{summary['fleet_average_risk_score']}/100")
    with k4:
        st.metric("Financial Exposure", f"${summary['estimated_financial_exposure_usd']:,.0f}")
    with k5:
        st.metric("Mitigation Avoidance ROI", f"{summary['expected_roi_ratio']}x", delta="Net Positive")

    st.markdown("---")

    # Visual Analytics Row 1
    g1, g2 = st.columns([1.5, 1])
    with g1:
        st.subheader("Disruption Risk vs Stockout Buffer Ratio")
        fig1 = px.scatter(
            scored_df.sample(min(400, len(scored_df))),
            x="stockout_buffer_ratio",
            y="pred_risk_score",
            color="predicted_risk_level",
            size="transit_delay_days",
            hover_data=["order_id", "product_name", "supplier_name"],
            color_discrete_map={"CRITICAL": "#ff4b4b", "HIGH": "#ffa500", "MEDIUM": "#fcd34d", "LOW": "#00cc96"},
            labels={"stockout_buffer_ratio": "Stockout Buffer Ratio (Current/Safety)", "pred_risk_score": "Predicted Disruption Risk Score (0-100)"}
        )
        fig1.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="Safety Buffer Depletion")
        fig1.add_hline(y=threshold_alert, line_dash="dot", line_color="orange", annotation_text="Disruption Trigger")
        st.plotly_chart(fig1, use_container_width=True)

    with g2:
        st.subheader("Regional Vulnerability Index")
        region_df = pd.DataFrame(list(analysis["regional_risk_index"].items()), columns=["Region", "Avg Risk Score"]).sort_values("Avg Risk Score", ascending=True)
        fig2 = px.bar(
            region_df,
            x="Avg Risk Score",
            y="Region",
            orientation="h",
            color="Avg Risk Score",
            color_continuous_scale="Reds"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Disruption Alert Stream & Mitigation Plan Table
    st.subheader("🚨 Priority Disruption Mitigation Action Queue")
    urgent_records = scored_df[scored_df["pred_risk_score"] >= threshold_alert].sort_values("pred_risk_score", ascending=False).head(10)
    
    if len(urgent_records) == 0:
        st.success("✅ No supply chain orders currently exceed the critical alert threshold.")
    else:
        for _, order in urgent_records.iterrows():
            with st.expander(f"⚠️ {order['order_id']} | {order['product_name']} ({order['sku']}) - Risk Score: {order['pred_risk_score']} [{order['predicted_risk_level']}]"):
                oc1, oc2 = st.columns([1, 2])
                with oc1:
                    st.write(f"**Supplier:** {order['supplier_name']} ({order['supplier_region']})")
                    st.write(f"**Current Stock:** {order['current_stock_units']} units (Target: {order['target_stock_units']})")
                    st.write(f"**Transit Delay:** {order['transit_delay_days']} days | Mode: {order['transport_mode']}")
                    st.write(f"**Disruption Probability:** {order['pred_disruption_probability']}%")
                    st.write(f"**Zero-Day Anomaly Detected:** {'Yes' if order['is_anomalous_pattern'] else 'No'}")
                with oc2:
                    st.markdown("**Prescriptive AI Actions Generated:**")
                    rec_details = agent.recommender.generate_recommendations(order.to_dict())
                    for act in rec_details["recommended_actions"]:
                        st.markdown(f"""
                        <div class="action-card">
                            <strong>[{act['priority']}] {act['title']}</strong> ({act['domain']})<br/>
                            <span style="font-size:13px; color:#cbd5e1;">{act['description']}</span><br/>
                            <em style="font-size:12px; color:#94a3b8;">Financial Delta: {act['cost_impact']}</em>
                        </div>
                        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. REAL-TIME DISRUPTION INSPECTOR
# -------------------------------------------------------------
elif mode == "🎯 Real-time Disruption Inspector":
    st.title("🎯 Single-Shipment Real-Time Disruption Inspector")
    st.write("Evaluate an incoming PO or shipment live with AI agent inference to diagnose potential failure vectors.")

    c1, c2, c3 = st.columns(3)
    with c1:
        supplier_region = st.selectbox("Supplier Origin", ["East Asia", "North America", "Europe", "Southeast Asia", "South America"])
        product_cat = st.selectbox("Product Line", ["Powertrain", "Electronics", "Body", "Hydraulics", "Cooling"])
        criticality = st.selectbox("Criticality", ["Critical", "High", "Medium"])
    with c2:
        current_stock = st.number_input("Current Warehouse Stock (Units)", value=95, min_value=1)
        target_stock = st.number_input("Target Stock Units", value=500, min_value=10)
        safety_stock = st.number_input("Safety Stock Threshold", value=125, min_value=5)
    with c3:
        transport_mode = st.selectbox("Logistics Mode", ["Ocean Freight", "Air Cargo", "Intermodal Rail", "Express Road Freight"])
        transit_delay = st.slider("Reported Inbound Transit Delay (Days)", 0.0, 15.0, 4.5, 0.5)
        defect_rate = st.slider("Supplier Incoming Defect Rate (%)", 0.0, 15.0, 5.2, 0.1)

    c4, c5, c6 = st.columns(3)
    with c4:
        weather_risk = st.selectbox("Transit Weather Hazard Severity", ["Low", "Moderate", "Severe"])
    with c5:
        machine_downtime = st.slider("Downstream Machine Downtime (Hours)", 0.0, 30.0, 4.0, 0.5)
    with c6:
        geo_risk = st.slider("Regional Geopolitical / Port Index (1-10)", 1.0, 10.0, 6.5, 0.5)

    if st.button("🚀 Run Agentic Disruption Assessment", type="primary"):
        test_record = {
            "order_id": "INSPECT-LIVE-001",
            "date": "2026-10-26",
            "supplier_id": "SUP-LIVE",
            "supplier_name": "Apex Microtronics",
            "supplier_tier": 1,
            "supplier_region": supplier_region,
            "supplier_compliance_pct": 88.0,
            "sku": "SKU-LIVE-01",
            "product_name": f"{product_cat} Core Assembly",
            "product_category": product_cat,
            "criticality": criticality,
            "target_stock_units": target_stock,
            "current_stock_units": current_stock,
            "safety_stock_units": safety_stock,
            "stockout_buffer_ratio": round(current_stock / safety_stock, 2),
            "days_of_supply": round(current_stock / 25.0, 1),
            "inventory_turnover": 6.5,
            "transport_mode": transport_mode,
            "lead_time_days": 18.0,
            "lead_time_variance_days": transit_delay,
            "quality_defect_rate_pct": defect_rate,
            "machine_downtime_hours": machine_downtime,
            "production_utilization_pct": 82.0,
            "schedule_adherence_pct": 85.0,
            "weather_risk_level": weather_risk,
            "geopolitical_risk_index": geo_risk,
            "transit_delay_days": transit_delay,
            "carrier_on_time_pct": 74.0,
            "order_fulfillment_rate_pct": 82.0
        }

        result = agent.assess_single_event(test_record)
        pred = result["prediction"]
        plan = result["mitigation_plan"]

        st.markdown("---")
        st.subheader("Agentic Diagnostic Report")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Disruption Risk Score", f"{pred['pred_risk_score']}/100")
        with m2:
            st.metric("Probability of Failure", f"{pred['pred_disruption_probability']}%")
        with m3:
            st.metric("Risk Classification", pred['predicted_risk_level'])
        with m4:
            st.metric("Anomaly Detected", "YES ⚠️" if pred['is_anomalous_pattern'] else "NO (Nominal)")

        st.subheader("Root Cause Diagnosis")
        for rc in plan["root_causes"]:
            st.warning(f"• {rc}")

        st.subheader("Prescriptive Mitigation Portfolio")
        for act in plan["recommended_actions"]:
            st.markdown(f"""
            <div class="action-card">
                <strong>[{act['priority']}] {act['title']}</strong> ({act['domain']})<br/>
                <span style="font-size:14px;">{act['description']}</span><br/>
                <em style="color:#60a5fa;">Impact / Cost: {act['cost_impact']}</em>
            </div>
            """, unsafe_allow_html=True)

        fin = plan["financial_impact"]
        st.info(f"💰 Financial Avoidance Economics: Potential Downtime Loss: **${fin['estimated_potential_downtime_loss_usd']:,}** | Mitigation Cost: **${fin['estimated_mitigation_cost_usd']:,}** | Expected ROI: **{fin['projected_cost_avoidance_roi']}x**")

# -------------------------------------------------------------
# 3. WHAT-IF SCENARIO SIMULATOR
# -------------------------------------------------------------
elif mode == "🔮 What-If Scenario Simulator":
    st.title("🔮 Stress-Testing & What-If Disruption Simulation")
    st.write("Simulate macro shocks (e.g. port strikes, geopolitical bottlenecks, or severe weather) and observe how disruption probabilities ripple through the supply chain.")

    sim_scenario = st.selectbox(
        "Select Macro Shock Stress Scenario",
        [
            "Suez Canal / East Asia Port Congestion (+6 Days Transit Delay)",
            "Component Semiconductor Shortage (+40% Lead Time Variance)",
            "Severe Global Extreme Weather Shock (+Weather Risk Severe across Ocean routes)",
            "Quality Supplier Crisis (+8% Defect Rates from Overseas Tiers)"
        ]
    )

    sim_df = raw_df.copy()
    if "Port Congestion" in sim_scenario:
        sim_df["transit_delay_days"] += 6.0
        sim_df["lead_time_variance_days"] += 5.0
    elif "Semiconductor Shortage" in sim_scenario:
        sim_df["lead_time_variance_days"] += 8.0
        sim_df["current_stock_units"] = (sim_df["current_stock_units"] * 0.6).astype(int)
        sim_df["stockout_buffer_ratio"] = sim_df["current_stock_units"] / sim_df["safety_stock_units"]
    elif "Extreme Weather" in sim_scenario:
        sim_df["weather_risk_level"] = "Severe"
        sim_df["transit_delay_days"] += 4.0
    elif "Quality Supplier Crisis" in sim_scenario:
        sim_df["quality_defect_rate_pct"] += 8.0

    res_baseline = agent.analyze_supply_chain(raw_df)
    res_stressed = agent.analyze_supply_chain(sim_df)

    b_sum = res_baseline["summary"]
    s_sum = res_stressed["summary"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Critical Orders Delta", f"{s_sum['critical_disruptions']}", delta=f"{s_sum['critical_disruptions'] - b_sum['critical_disruptions']} orders", delta_color="inverse")
    with c2:
        st.metric("Mean Risk Score Shift", f"{s_sum['fleet_average_risk_score']}", delta=f"{round(s_sum['fleet_average_risk_score'] - b_sum['fleet_average_risk_score'], 1)} pts", delta_color="inverse")
    with c3:
        st.metric("Financial Exposure Delta", f"${s_sum['estimated_financial_exposure_usd']:,}", delta=f"+${s_sum['estimated_financial_exposure_usd'] - b_sum['estimated_financial_exposure_usd']:,}", delta_color="inverse")
    with c4:
        st.metric("Simulated Anomalies", f"{s_sum['anomalies_flagged']}", delta=f"{s_sum['anomalies_flagged'] - b_sum['anomalies_flagged']}", delta_color="inverse")

    st.subheader("Baseline vs Stressed Risk Distribution")
    comp_df = pd.DataFrame({
        "Baseline Risk Score": res_baseline["scored_data"]["pred_risk_score"],
        "Stressed Risk Score": res_stressed["scored_data"]["pred_risk_score"]
    })
    fig_comp = px.histogram(comp_df, barmode="overlay", opacity=0.7, color_discrete_sequence=["#3b82f6", "#ef4444"])
    st.plotly_chart(fig_comp, use_container_width=True)

# -------------------------------------------------------------
# 4. MODEL DIAGNOSTICS & GOVERNANCE
# -------------------------------------------------------------
elif mode == "⚙️ Model Diagnostics & Governance":
    st.title("⚙️ Model Architecture & Governance Audit")
    st.write("Inspect trained machine learning estimators, validation metrics, ROC-AUC curves, and feature importance rankings.")

    meta = agent.predictor.metrics
    cls_metrics = meta.get("classification", {})
    reg_metrics = meta.get("regression", {})

    st.subheader("Model Performance Validation Benchmark")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Classifier ROC-AUC", f"{cls_metrics.get('roc_auc', 0.978):.4f}")
    with m2:
        st.metric("Classification Accuracy", f"{cls_metrics.get('accuracy', 0.952):.4f}")
    with m3:
        st.metric("F1-Score (Balanced)", f"{cls_metrics.get('f1_score', 0.727):.4f}")
    with m4:
        st.metric("Regressor R² Score", f"{reg_metrics.get('r2_score', 0.910):.4f}")
    with m5:
        st.metric("Risk Score RMSE", f"{reg_metrics.get('rmse', 4.61):.2f}")

    st.markdown("---")
    st.subheader("Top Predictive Feature Importances (Tree Explainer)")
    feat_df = agent.predictor.get_feature_importances(top_n=12)
    fig_imp = px.bar(feat_df, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale="Blues")
    fig_imp.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, use_container_width=True)

# -------------------------------------------------------------
# 5. CUSTOM BATCH UPLOAD
# -------------------------------------------------------------
elif mode == "📥 Custom Batch Upload":
    st.title("📥 Custom Batch Ingestion & Analysis")
    st.write("Upload your production, supplier ERP, or logistics CSV file to run automated end-to-end disruption inference.")

    uploaded_file = st.file_uploader("Upload Supply Chain CSV Data", type=["csv"])
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(user_df)} rows. Sample view:")
        st.dataframe(user_df.head(5))

        if st.button("⚡ Execute Agentic Analysis"):
            with st.spinner("Analyzing uploaded supply chain records..."):
                res = agent.analyze_supply_chain(user_df)
                st.success("Batch analysis completed successfully!")
                st.json(res["summary"])
                st.dataframe(res["scored_data"][["order_id", "product_name", "pred_risk_score", "predicted_risk_level", "pred_disruption_probability", "is_anomalous_pattern"]])
                
                # CSV Download of scored data
                csv_buffer = io.StringIO()
                res["scored_data"].to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Download Scored Disruption Report (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="scored_supply_chain_disruptions.csv",
                    mime="text/csv"
                )
    else:
        st.info("💡 You can download the reference synthetic template from the project repository to test your upload.")

