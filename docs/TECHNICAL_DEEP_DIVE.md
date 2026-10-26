# Technical Deep Dive: Architecture, Models, and Algorithms

This technical deep dive details the architecture, feature engineering formulations, statistical assumptions, and modeling choices underlying the **NexChain AI** operations system.

---

## 1. System Architecture

NexChain AI is structured as a decoupled, layered AI operations engine:

```
[ Ingest Stream (ERP / Logistics / Warehouse) ]
                      │
                      ▼
┌───────────────────────────────────────────────┐
│        Feature Engineering & Preprocessor      │
│  - RobustScaler (outlier resilience)          │
│  - Temporal cyclical encoding                 │
│  - Stockout Buffer Ratio & Vulnerability Index│
└──────────────────────┬────────────────────────┘
                       │ Normalized Vector (36-dim)
                       ▼
┌───────────────────────────────────────────────┐
│         Multi-Task Predictive Ensemble        │
│  ├── Disruption Classifier (Random Forest)    │
│  ├── Continuous Risk Regressor (Grad. Boost)  │
│  └── Anomaly Detector (Isolation Forest)      │
└──────────────────────┬────────────────────────┘
                       │ Predictions & Anomaly Tags
                       ▼
┌───────────────────────────────────────────────┐
│         Prescriptive Recommendation Engine    │
│  ├── Dynamic Sourcing Allocation (Dual-source)│
│  ├── Dynamic Buffer Recalibration             │
│  ├── Expedited Freight Mode Arbitrage         │
│  └── Assembly Line Throughput Balancing       │
└──────────────────────┬────────────────────────┘
                       │ Action Plans & Financial Loss Avoidance
                       ▼
       [ Streamlit Interactive Command Center ]
```

---

## 2. Engineered Supply Chain Metrics & Formulations

Supply chain failures rarely stem from a single variable in isolation; rather, they arise from compound bottlenecks. NexChain AI formulates specific domain metrics:

### 1. Stockout Buffer Ratio ($SBR$)
Measures the safety margin of inventory on hand relative to the designated minimum threshold:
$$SBR = \frac{I_{\text{current}}}{I_{\text{safety}}}$$
- When $SBR < 1.0$, the warehouse has breached the buffer zone and entered production starvation risk.

### 2. Stock Deficit Percentage ($SDP$)
Quantifies the shortfall against optimal target stock:
$$SDP = \max\left(0, \frac{I_{\text{target}} - I_{\text{current}}}{I_{\text{target}}}\right) \times 100$$

### 3. Logistics Vulnerability Index ($LVI$)
A composite index reflecting compounded delays, quality defects, and depleted safety margins:
$$LVI = (2.0 \cdot \Delta_{\text{transit}}) + Q_{\text{defect}} + \frac{10.0}{\max(SBR, 0.1)}$$
Where:
- $\Delta_{\text{transit}}$ is the transit delay in days.
- $Q_{\text{defect}}$ is the percentage of defective parts upon receipt.
- $SBR$ is the Stockout Buffer Ratio.

---

## 3. Machine Learning Ensemble Architecture

### Model 1: Disruption Classifier (Random Forest)
- **Objective:** Predict binary event $\hat{y} \in \{0, 1\}$ indicating whether an order will fail to meet scheduled production requirements.
- **Algorithm:** Random Forest with 150 estimators, balanced class weights, max depth 12.
- **Why Random Forest:** Robust to non-linear interactions between categorical supplier regions and numerical delay variance without overfitting.
- **Validation Metrics:**
  - Accuracy: **95.29%**
  - ROC-AUC: **0.9788**
  - Precision: **84.62%**
  - F1-Score: **72.73%**

### Model 2: Continuous Disruption Risk Score Regressor (Gradient Boosting)
- **Objective:** Estimate a continuous Risk Severity Index $\hat{S} \in [0, 100]$.
- **Algorithm:** Gradient Boosting Regressor (120 estimators, learning rate 0.08, max depth 5).
- **Validation Metrics:**
  - R² Score: **0.9107**
  - Root Mean Squared Error (RMSE): **4.617**
  - Mean Absolute Error (MAE): **3.527**

### Model 3: Unsupervised Zero-Day Anomaly Detector (Isolation Forest)
- **Objective:** Detect novel disruption patterns that do not conform to historical training distribution (e.g. cascading supply chain shocks or unprecedented lead-time spikes).
- **Algorithm:** Isolation Forest (100 isolation trees, contamination rate 8%).
- **Scoring:** Produces an anomaly decision flag along with a continuous anomaly severity score.

---

## 4. Prescriptive Action Optimization

The recommendation engine translates predictive outputs into financial and operational actions:

1. **Dual-Sourcing Heuristic:** When $S \ge 45$ or defect rate $\ge 4.5\%$, the system retrieves secondary suppliers from the knowledge catalog with minimum lead time and reliability $\ge 90\%$, recommending a 35% PO reallocation.
2. **Buffer Recalibration:** When $SBR < 1.0$, safety stock threshold is automatically scaled up by $25\%$ of target capacity.
3. **Logistics Rerouting Arbitrage:** If transit delay $\ge 2.0$ days on ocean freight for critical assembly components, the engine calculates the cost difference between line downtime losses vs expedited air cargo fees (\$4,500 - \$8,000), presenting the net ROI to managers.

