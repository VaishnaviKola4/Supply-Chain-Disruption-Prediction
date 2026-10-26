# NexChain AI: Manufacturing Supply Chain Disruption Prediction & Autonomous Mitigation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30%2B-orange.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Problem Statement

Global manufacturing supply chains operate under tight tolerances where unexpected disruptions can cascade into catastrophic line stoppages and financial penalties. Today's industrial supply networks face critical operational vulnerabilities:

- **Lack of Multi-Tier Visibility:** Traditional ERP and warehouse management systems operate in isolated silos, lacking unified visibility across tier-1/tier-2 suppliers, transit corridors, and plant floor consumption.
- **Reactive Disruption Response:** Conventional operations react only *after* a stockout occurs or a shipment fails to arrive, leading to emergency expedited freight surcharges, unplanned factory downtime (costing upwards of \$50,000–\$150,000 per hour in automotive and electronics plants), and unfulfilled customer commitments.
- **Unprecedented Shocks & Volatility:** Macro headwinds—including maritime port congestion, geopolitical bottlenecks, extreme climate events, and volatile supplier compliance—demand predictive foresight rather than static heuristic thresholds.
- **Absence of Prescriptive Guidance:** Even when risk dashboards flag potential bottlenecks, supply chain managers lack autonomous, quantified mitigation playbooks (e.g., dual-sourcing allocation, dynamic safety buffer recalibration, freight mode arbitrage) accompanied by actionable ROI evaluations.

**NexChain AI** directly addresses these challenges by delivering an intelligent, predictive, and prescriptive AI operations agent that continuously monitors supply chain telemetry, preempts impending failures with machine learning and anomaly detection, and autonomously formulates prioritized mitigation directives.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Project Overview and Purpose](#project-overview-and-purpose)
3. [Repository Structure](#repository-structure)
4. [Algorithm Overview](#algorithm-overview)
5. [Core Features Explanation](#core-features-explanation)
6. [Model Architecture (High-Level)](#model-architecture-high-level)
7. [Data Requirements and Format](#data-requirements-and-format)
8. [Installation and Setup Instructions](#installation-and-setup-instructions)
9. [Usage Guide](#usage-guide)
10. [Documentation Folder](#documentation-folder)
11. [Contribution Guidelines](#contribution-guidelines)
12. [License Information](#license-information)

---

## Project Overview and Purpose

Modern manufacturing relies on agile supply chains where single points of failure can halt production. 

**NexChain AI** provides an end-to-end autonomous agentic solution that:
- Ingests telemetry across suppliers, warehouse stock, production schedules, and logistics.
- Employs a multi-task machine learning ensemble to classify disruption events, forecast continuous risk scores ($0 - 100$), and detect zero-day sensor anomalies.
- Generates prescriptive mitigation action plans covering dynamic dual-sourcing, automated safety stock recalibration, expedited freight switching, and production line rebalancing with quantified financial avoidance ROI.

---

## Repository Structure

```
Supply Chain Disruption Prediction/
├── data/
│   └── supply_chain_data.csv            # Enterprise synthetic dataset (3,500 multi-tier records)
├── models/
│   ├── preprocessor.joblib              # Fitted RobustScaler & categorical feature pipeline
│   ├── classifier.joblib                # Random Forest Disruption Classifier (ROC-AUC: 0.9788)
│   ├── regressor.joblib                 # Gradient Boosting Disruption Risk Regressor (R²: 0.9107)
│   ├── anomaly_detector.joblib          # Isolation Forest Anomaly Detector
│   └── metrics.joblib                   # Persisted validation benchmarks
├── src/
│   ├── __init__.py                      # Package entry point
│   ├── data_generator.py                # Synthetic multi-tier supply chain data generator
│   ├── preprocessor.py                  # Domain feature engineering (SBR, LVI, robust scaling)
│   ├── predictor.py                     # Multi-task predictive ML engine
│   ├── recommendation_engine.py         # Multi-domain prescriptive mitigation engine
│   ├── agent.py                         # Autonomous orchestrator (SupplyChainAgent)
│   └── app.py                           # 5-console Streamlit interactive UI command center
├── docs/
│   ├── USE_CASES.md                     # Scenario playbooks (automotive seasonal, scrap reduction)
│   ├── HOW_TO_USE.md                    # Setup, training, execution, and UI operating guide
│   ├── API_INTERFACE_GUIDE.md           # Python classes, method signatures, and data contracts
│   └── TECHNICAL_DEEP_DIVE.md           # Math formulations, ML algorithms, and heuristics
├── tests/
│   └── test_pipeline.py                 # Comprehensive unit & integration test suite (Pytest)
├── README.md                            # Primary documentation following reference repository standard
├── requirements.txt                     # Pinned project dependencies
└── LICENSE                              # MIT License
```

---

## Algorithm Overview

NexChain AI integrates statistical feature engineering, supervised classification, gradient boosted regression, and unsupervised anomaly detection into a coordinated operational intelligence pipeline:

```
Telemetry Stream (Suppliers, Inventory, Factory, Freight)
                        │
                        ▼
      ┌────────────────────────────────────┐
      │   Domain Feature Transformations   │
      │   • SBR (Stockout Buffer Ratio)    │
      │   • LVI (Vulnerability Index)      │
      │   • RobustScaler Normalization     │
      └─────────────────┬──────────────────┘
                        │ Normalized 36-dimensional feature vector
       ┌────────────────┼────────────────┐
       ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Random      │ │  Gradient    │ │  Isolation   │
│  Forest      │ │  Boosting    │ │  Forest      │
│  Classifier  │ │  Regressor   │ │  Detector    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │ Event Flag     │ Score (0-100)  │ Novel Outlier
       └────────────────┼────────────────┘
                        ▼
      ┌────────────────────────────────────┐
      │   Prescriptive Action Synthesis    │
      │   • Dual-Sourcing Allocation       │
      │   • Dynamic Safety Stock Elevation │
      │   • Freight Expediting Arbitrage   │
      │   • Assembly Line Throughput Tune  │
      └─────────────────┬──────────────────┘
                        ▼
      Executable Decisions & Financial ROI
```

### 1. Domain Feature Formulations
- **Stockout Buffer Ratio ($SBR$):**
  $$SBR = \frac{\text{Current Stock}}{\text{Safety Stock}}$$
  Identifies buffer boundary exhaustion ($SBR < 1.0$) before actual stockout takes place.
- **Logistics Vulnerability Index ($LVI$):**
  $$LVI = (2.0 \times \text{Transit Delay Days}) + \text{Defect Rate \%} + \left(\frac{10.0}{\max(SBR, 0.1)}\right)$$
  Captures compound risk exposure across supplier quality, delivery variance, and depleted buffer margins.

### 2. Supervised Disruption Classification (Random Forest)
- Predicts discrete failure probability $\hat{y} \in \{0, 1\}$.
- Utilizes an ensemble of 150 decorrelated decision trees with balanced subsample weights to handle imbalanced disruption occurrences.
- **Performance:** **ROC-AUC: 0.9788**, **Accuracy: 95.29%**, **Precision: 84.62%**.

### 3. Continuous Risk Score Estimation (Gradient Tree Boosting)
- Computes an exact Disruption Risk Severity Score $\hat{S} \in [0, 100]$.
- Sequentially fits regression trees using a pseudo-residual gradient loss function ($\text{learning rate} = 0.08, \text{estimators} = 120, \text{max depth} = 5$).
- **Performance:** **$R^2$: 0.9107**, **RMSE: 4.617**, **MAE: 3.527**.

### 4. Unsupervised Outlier & Zero-Day Anomaly Detection (Isolation Forest)
- Isolates multidimensional anomalous behavior without relying on historic target labels.
- Computes anomaly isolation scores via path-length distributions across 100 random sub-sampling trees, flagging zero-day macro shocks (e.g., sudden regional geopolitical embargos or unannounced supplier outages).

### 5. Multi-Domain Prescriptive Heuristics
- **Dual-Sourcing Routing:** Reallocates 35% of upcoming order volume to high-reliability backup suppliers when risk scores exceed 45 or defect rates exceed 4.5%.
- **Safety Stock Expansion:** Recalibrates buffer targets upward by +25% when $SBR < 1.0$.
- **Freight Mode Arbitrage:** Computes trade-off economics between prospective plant downtime losses vs. air-freight premium conversion.

---

## Core Features Explanation

- 🌐 **Multi-Tier Telemetry Ingestion & Feature Engineering:**
  Processes lead-time variance, supplier compliance, days-of-supply, transit delays, and computes domain metrics like the *Stockout Buffer Ratio* and *Logistics Vulnerability Index*.
- 🤖 **Multi-Task Predictive Ensemble:**
  - **Random Forest Disruption Classifier:** Flags high-risk shipments ($\text{ROC-AUC} = 0.9788, \text{Accuracy} = 95.29\%$).
  - **Gradient Boosting Risk Regressor:** Estimates continuous disruption severity scores ($R^2 = 0.9107, \text{RMSE} = 4.617$).
  - **Isolation Forest Anomaly Detector:** Unsupervised detector flagging novel zero-day disruptions and operational outliers.
- 💡 **Autonomous Recommendation Engine:**
  Delivers prioritized, actionable mitigation playbooks (dual-sourcing with vetted backup suppliers, buffer elevation, freight expediting) with explicit ROI projections.
- 📊 **Interactive Streamlit Command Center:**
  Features 5 dedicated consoles: Global Fleet Operations, Real-Time Disruption Inspector, What-If Macro Stress Simulator, Model Governance, and Batch CSV Ingestion.
- 🧪 **Enterprise Test Suite & Modularity:**
  Fully covered by unit and integration tests using `pytest` with modular, production-ready code structure.

---

## Model Architecture (High-Level)

```
 Raw Data Ingestion (Suppliers, Inventory, Production, Logistics)
                            │
                            ▼
           [ RobustScaler + Feature Engineering ]
      - Stockout Buffer Ratio = Current Stock / Safety Stock
      - Stock Deficit % & Logistics Vulnerability Index
                            │
                            ▼
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
  Random Forest Classifier            Gradient Boosting Regressor
  - Predicts disruption event (0/1)   - Continuous Risk Score (0-100)
  - ROC-AUC: 0.9788, Acc: 95.29%      - R²: 0.9107, RMSE: 4.617
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
            Unsupervised Isolation Forest
            - Flags Zero-Day Outlier Patterns
                            │
                            ▼
        [ Prescriptive Recommendation Engine ]
  ┌─────────────────────────────────────────────────────────┐
  │ 1. Supplier Allocation: Reroute POs to secondary source │
  │ 2. Inventory Policy: Recalibrate safety stock buffers   │
  │ 3. Logistics Rerouting: Ocean Freight to Air Cargo     │
  │ 4. Production Rebalancing: Throttle throughput / shifts │
  └─────────────────────────────────────────────────────────┘
```

---

## Data Requirements and Format

Input datasets should be provided in CSV format with the following core columns:

| Column Name | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `order_id` | String | Unique purchase/shipment identifier | `ORD-2025-10024` |
| `date` | Date (YYYY-MM-DD) | Scheduled order or PO date | `2025-06-15` |
| `supplier_id` / `supplier_name` | String | Supplier code and entity name | `Apex Microtronics` |
| `supplier_region` | String | Origin geography (`East Asia`, `Europe`, etc.) | `East Asia` |
| `supplier_compliance_pct` | Float | Historical vendor on-time compliance % | `92.4` |
| `sku` / `product_name` | String | Part SKU and descriptive name | `SKU-PRO-01` |
| `product_category` | String | Domain category (`Powertrain`, `Electronics`, etc.) | `Powertrain` |
| `criticality` | String | Assembly importance (`Critical`, `High`, `Medium`) | `Critical` |
| `target_stock_units` | Integer | Baseline optimal inventory target | `500` |
| `current_stock_units` | Integer | Units on-hand in warehouse | `85` |
| `safety_stock_units` | Integer | Designated safety threshold | `125` |
| `transport_mode` | String | Logistics mode (`Ocean Freight`, `Air Cargo`, etc.) | `Ocean Freight` |
| `lead_time_days` | Float | Quoted lead time in days | `18.0` |
| `transit_delay_days` | Float | Current transit delay variance in days | `4.5` |
| `quality_defect_rate_pct` | Float | Defective parts percentage | `5.2` |
| `machine_downtime_hours` | Float | Downstream assembly downtime hours | `3.5` |

---

## Installation and Setup Instructions

### 1. Clone or Open the Repository
```bash
git clone https://github.com/your-org/supply-chain-disruption-prediction.git
cd supply-chain-disruption-prediction
```

### 2. Set Up a Python Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Generate Synthetic Data
```bash
python src/data_generator.py
```

### 2. Train the Machine Learning Models
```bash
python -m src.predictor
```

### 3. Launch the Interactive Web Dashboard
```bash
streamlit run src/app.py
```
Open your web browser and navigate to `http://localhost:8501`.

### 4. Run Test Suite
```bash
python -m pytest tests/ -v
```

---

## Documentation Folder

Comprehensive architectural and user documentation is available in the [`docs/`](docs/) directory:

- 📖 [**Use Cases (`docs/USE_CASES.md`)**](docs/USE_CASES.md): Real-world manufacturing deployment scenarios including automotive seasonal surges, shelf-life scrap minimization, and port congestion management.
- 🛠️ [**How to Use (`docs/HOW_TO_USE.md`)**](docs/HOW_TO_USE.md): Step-by-step configuration, preprocessing pipeline execution, model retraining, and command center navigation.
- 🔌 [**API & Interface Guide (`docs/API_INTERFACE_GUIDE.md`)**](docs/API_INTERFACE_GUIDE.md): Developer reference for Python classes (`SupplyChainAgent`, `DisruptionRecommendationEngine`), method signatures, and data contracts.
- 🧠 [**Technical Deep Dive (`docs/TECHNICAL_DEEP_DIVE.md`)**](docs/TECHNICAL_DEEP_DIVE.md): Mathematical formulations for $SBR$ and $LVI$, classifier/regressor hyperparameters, validation metrics, and heuristic optimization rules.

---

## Contribution Guidelines

Contributions are welcomed! Follow these steps to contribute:

1. **Fork the Repository:** Create your own branch (`git checkout -b feature/dynamic-leadtime-enrichment`).
2. **Follow Coding Standards:** Adhere to PEP 8 conventions. Format using `black` and lint with `flake8`.
3. **Add Tests:** Ensure all new algorithms and features have corresponding unit tests in `tests/`.
4. **Verify Test Suite:** Run `python -m pytest tests/` and confirm 100% pass rate.
5. **Open a Pull Request:** Detail your changes, test results, and rationale in your PR description.

---

## License Information

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for full details.
