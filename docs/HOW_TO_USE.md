# Step-by-Step Guide: How to Use NexChain AI

This guide provides instructions for installing, configuring, training, and operating the NexChain AI Supply Chain Disruption Prediction system.

---

## 1. System Requirements

- **Operating System:** Windows 10/11, Linux (Ubuntu 20.04+), or macOS
- **Python:** Version 3.10 or 3.11 (Python 3.11 recommended)
- **RAM:** Minimum 4GB (8GB recommended for large batch simulations)
- **Storage:** ~500MB free disk space

---

## 2. Installation & Setup

### Step 1: Clone or Open the Repository
```bash
cd "Supply Chain Disruption Prediction"
```

### Step 2: Set up a Python Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On Linux/macOS:
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 3. Data Generation & Preprocessing

The system includes an automated enterprise synthetic data generator producing multi-tier supply chain records.

### Generate Synthetic Dataset:
```bash
python src/data_generator.py
```
This produces `data/supply_chain_data.csv` containing 3,500 records with realistic lead times, inventory levels, transit delays, and ground-truth disruption labels.

### Run Preprocessing & Feature Engineering Pipeline:
```bash
python src/preprocessor.py
```
This fits `RobustScaler` and one-hot categorical mappings, persisting the transformer into `models/preprocessor.joblib`.

---

## 4. Model Training & Validation

To train the machine learning ensemble (Random Forest Disruption Classifier, Gradient Boosting Risk Regressor, and Isolation Forest Anomaly Detector):

```bash
python -m src.predictor
```

**Expected Training Outputs:**
- Preprocessor: `models/preprocessor.joblib`
- Classification Model: `models/classifier.joblib` (ROC-AUC ~ 0.978)
- Regression Model: `models/regressor.joblib` (R² ~ 0.910, RMSE ~ 4.6)
- Anomaly Detector: `models/anomaly_detector.joblib`
- Performance Metrics: `models/metrics.joblib`

---

## 5. Running the Interactive Streamlit UI

Launch the web command center:

```bash
streamlit run src/app.py
```
Once launched, open your browser at `http://localhost:8501`.

### UI Console Sections:
1. **Fleet Operations Dashboard:** Filter by category, origin region, or component criticality; monitor fleet KPIs, risk scatter charts, and the high-priority action queue.
2. **Real-time Disruption Inspector:** Enter custom shipment/PO parameters (stock, transit delays, defect rate) and click **Run Agentic Disruption Assessment** to receive immediate diagnosis and prescriptive actions.
3. **What-If Scenario Simulator:** Simulate stress shocks (port congestion, chip shortages, severe weather) and quantify fleet risk score shifts and dollar loss exposure.
4. **Model Diagnostics & Governance:** Inspect classifier ROC-AUC, regressor RMSE, and tree-based feature importance rankings.
5. **Custom Batch Upload:** Upload any enterprise CSV dataset to score records and download augmented disruption intelligence reports.

---

## 6. Running Automated Tests

Run the complete test suite using `pytest`:

```bash
python -m pytest tests/ -v
```
All unit tests should pass with 100% coverage across data generation, preprocessing, model inference, recommendation logic, and agent orchestration.

