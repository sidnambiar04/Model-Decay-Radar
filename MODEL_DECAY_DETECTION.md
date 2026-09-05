# Model Decay Radar: Architecture, Achievements, Algorithms, and UI Guide

This document provides a comprehensive, production-grade guide to the **Model Decay Radar** system. It details the **achievements completed to date**, **project features and capabilities**, a complete breakdown of **all algorithms**, an overview of the **Next.js telemetry UI**, and the **roadmap for production deployment**.

---

## 1. Achievements Completed Till Now

The project has achieved a fully functional, end-to-end model monitoring and drift detection MVP, verified across **16 comprehensive test stages**:

- [x] **Stage 1: Reference Baseline & Scaler Integration** — Automated Min-Max feature scaling fitted on historical reference data (`synthetic_swat.csv`).
- [x] **Stage 2: Asynchronous Real-time Prediction API** — FastAPI inference server with `/predict` endpoint, unblocked non-blocking in-memory ingestion buffering, and background thread orchestration.
- [x] **Stage 3: Deep Autoencoder & KL Permutation Test** — PyTorch Variational Autoencoder (VAE) reconstruction loss tracking combined with a 1,000-shuffle Permutation Test for exact statistical $p$-value calculation.
- [x] **Stage 4: Distance Metrics & Univariate Shift** — Wasserstein distance (Earth Mover's Distance) and per-feature 2-sample Kolmogorov-Smirnov (KS) statistical test implementation.
- [x] **Stage 5: Streaming Drift Detectors** — ADWIN (Adaptive Windowing) and DDM (Drift Detection Method) integration for continuous streaming drift detection.
- [x] **Stage 6: Epistemic Uncertainty Estimation** — 3-member LSTM RNN Ensemble utilizing Monte Carlo (MC) Dropout (10 stochastic forward passes per sample) to quantify model variance/novelty.
- [x] **Stage 7: Drift Signal Fusion Engine** — Multi-detector signal aggregator fusing VAE MSE, KL $p$-value, KS ratio, Wasserstein distance, ADWIN, and DDM into a composite score and categorizing drift types (`gradual`, `sudden`, `covariate_drift`, `concept_drift`, `none`).
- [x] **Stage 8: Model Health Score (MHS) Fusion** — Weighted model health score calculation combining Accuracy, Drift, Uncertainty, and Stability into `Healthy`, `Warning`, and `Critical` operational states.
- [x] **Stage 9: Side-by-Side Root Cause Analysis** — SHAP (KernelExplainer) feature attributions on VAE reconstruction error paired side-by-side with per-feature KS statistical shift results.
- [x] **Stage 10: Multi-Classifier Architecture** — Interchangeable classifier backend supporting Random Forest, XGBoost / Gradient Boosting, Multi-Layer Perceptron (MLP), and Logistic Regression.
- [x] **Stage 11: Class Imbalance-Aware Adaptive Retraining** — Borderline-SMOTE oversampling pipeline triggering automated fine-tuning when Model Health Score enters `Critical` state.
- [x] **Stage 12: Automated Pre-Deployment Validation Gate** — Safety evaluation gate comparing candidate retrained models against active baseline models before promotion.
- [x] **Stage 13: Versioned Model Registry with SQLite** — SQLite database (`models/model_registry.db`) for tracking model versions (`v1`, `v2`, ...), hyperparams, evaluation metrics, binary model artifacts, and active status.
- [x] **Stage 14: Interactive Real-time Drift Simulator** — Automated simulation runner injecting ramp drift, step drift, sensor noise, feature corruption, and simulating delayed ground-truth label arrivals.
- [x] **Stage 15: Operational Modes Support** — Seamless toggle between Production Mode (unlabelled streaming inference) and Evaluation/Demo Mode (labelled streams).
- [x] **Stage 16: Next.js Observability UI & Real-Time Telemetry Dashboard** — Modern dark glassmorphic dashboard built with Next.js, featuring live polling, dynamic SVG charts, SHAP/KS root-cause tabs, manual retraining triggers, and admin configuration controls.

---

## 2. Project Features & What the Project Can Do

The Model Decay Radar acts as an intelligent, automated monitoring system for deployed machine learning models. Key features include:

1. **Silent Data Drift Detection**: Detects gradual or sudden distribution shifts in complex, high-dimensional input streams (e.g. 51 IoT sensor metrics) *before* prediction degradation impacts downstream operations.
2. **Fast, Non-Blocking Inference**: Serves real-time predictions immediately via `POST /predict`. Input payloads are asynchronously buffered and evaluated in background monitoring cycles without adding latency to the inference API.
3. **Multi-Detector Signal Fusion**: Eliminates false positives by combining multiple statistical metrics (reconstruction loss, KL divergence, Wasserstein distance, per-feature KS tests, ADWIN, and DDM) into a unified composite drift index.
4. **Epistemic Uncertainty Tracking**: Distinguishes between out-of-distribution data novelty (model unfamiliarity) and simple random noise using a Monte Carlo Dropout RNN ensemble.
5. **Dynamic Model Health Score (MHS)**: Provides a single composite metric (0 to 1) representing overall model reliability based on accuracy, drift level, uncertainty, and output stability.
6. **Explainable Root Cause Attribution**: Highlights the specific features responsible for data drift using side-by-side SHAP attributions and per-feature Kolmogorov-Smirnov statistical tests.
7. **Imbalance-Aware Automated Retraining**: Automatically oversamples minority class anomalies using Borderline-SMOTE and fine-tunes models when health drops to `Critical`.
8. **Pre-Deployment Safety Validation Gate**: Validates retrained models against active production baselines, promoting models only if performance metrics exceed safety thresholds.
9. **SQLite Model Registry & Artifact Management**: Maintains model lineage, logs evaluation metrics, stores binary artifacts, and enables seamless rollbacks to previous versions.
10. **Interactive Drift Simulator & Scenario Runner**: Allows operators to inject custom drift profiles (ramp drift, step drift, sensor noise, corruption) to benchmark system responsiveness.
11. **Real-Time Telemetry UI**: Modern dark-themed dashboard presenting live gauges, dual SVG time-series charts, root-cause attribution tabs, manual action controls, and admin configuration modals.

---

## 3. Comprehensive Breakdown of Algorithms

| Algorithm / Module | Implementation & Mechanics | Operational Purpose |
| :--- | :--- | :--- |
| **Variational Autoencoder (VAE)** | PyTorch neural network compressing 51 features into a latent bottleneck and reconstructing them. Calculates MSE and KL reconstruction losses. | **Raw Distribution Drift Detection**. High reconstruction error signifies that incoming data deviates from the baseline distribution. |
| **Isotonic Regression Calibration** | Fits a monotonic, non-parametric regression mapping raw MSE reconstruction errors to a normalized [0, 1] scale using reference data. | **Score Normalization**. Maps arbitrary MSE values into calibrated drift probabilities. |
| **Permutation KL Divergence Test** | Non-parametric test shuffling baseline and current batch scores 1,000 times to compute exact empirical $p$-values. | **Statistical Significance**. Confirms data drift when $p < 0.01$, filtering out random noise. |
| **Kolmogorov-Smirnov (KS) 2-Sample Test** | Compares empirical cumulative distribution functions (eCDFs) of baseline vs. current batch per feature. | **Univariate Feature Drift**. Identifies which specific input features have statistically shifted ($p < 0.05$). |
| **Wasserstein Distance (Earth Mover's Distance)** | Calculates the minimum cost of transforming the current batch reconstruction score distribution into the reference distribution. | **Distribution Shift Magnitude**. Quantifies the physical distance between data distributions. |
| **ADWIN (Adaptive Windowing)** | Maintains an adaptive sliding window of prediction error rates, automatically splitting windows when statistical variance changes. | **Streaming Concept Drift**. Detects abrupt changes in streaming data environments. |
| **DDM (Drift Detection Method)** | Tracks error rates and standard deviation assuming binomial distribution to trigger Warning and Drift levels. | **Early Warning Trigger**. Signals early performance degradation in streaming predictions. |
| **RNN (LSTM) Ensemble + MC Dropout** | 3-member LSTM network ensemble executing 10 stochastic forward passes per sample with active dropout. | **Epistemic Uncertainty Quantification**. Measures model unfamiliarity and epistemic variance on time-series inputs. |
| **Borderline-SMOTE** | Oversampling algorithm identifying minority class samples near decision boundaries and generating synthetic samples. | **Class Imbalance Handling**. Prevents retraining bias when anomalies represent a tiny fraction of data. |
| **SHAP (KernelExplainer)** | Perturbs input features to compute Shapley values against VAE reconstruction error. | **Explainable Root Cause Attribution**. Pinpoints exact sensors driving the reconstruction anomaly. |
| **Drift Signal Fusion Engine** | Weighted signal aggregator combining VAE, KL, KS, Wasserstein, ADWIN, and DDM outputs into a composite drift index. | **Multi-Signal Synthesis & Classification**. Categorizes drift into `gradual`, `sudden`, `covariate_drift`, `concept_drift`, or `none`. |
| **Model Health Score (MHS) Engine** | Fuses weighted indicators: $\text{MHS} = 0.35 \cdot \text{Acc} + 0.25 \cdot (1-\text{Drift}) + 0.20 \cdot (1-\text{Uncertainty}) + 0.20 \cdot \text{Stability}$. | **Executive Health Monitoring**. Maps health into `Healthy` ($>0.85$), `Warning` ($0.65-0.85$), or `Critical` ($<0.65$). |
| **Validation Gate Engine** | Evaluates candidate retrained model against active baseline on validation set across Accuracy, F1-Score, Precision, and Recall. | **Pre-Deployment Safety Gate**. Promotes candidates only if safety thresholds are met; rejects regressions. |
| **Model Registry Engine** | SQLite-backed registry (`models/model_registry.db`) managing versions (`v1`, `v2`, ...), binary blobs, and metadata. | **Model Lineage & Governance**. Tracks version history, evaluation metrics, and handles promotion/rollback. |

---

## 4. Telemetry UI Breakdown (Next.js Dashboard)

The Next.js frontend (`frontend/app/page.tsx`) provides a real-time observability control center built with dark-mode aesthetic styling:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              MODEL DECAY RADAR DASHBOARD                               │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Header: [ API Status: Connected ] [ Version: v1 ] [ MHS: Healthy ] [ Alert: Nominal ]  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Executive KPI Cards:                                                                  │
│  [ Model Health Score ]     [ Drift Severity ]     [ Uncertainty ]     [ Accuracy ]    │
│     0.94 (Healthy)            0.02 (Nominal)         0.08 (Low)          94.5% (v1)     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Live Telemetry Charts (Custom SVG):                                                    │
│  • Chart 1: Reconstruction Error (MSE) vs. Dynamic Threshold over Time                  │
│  • Chart 2: Statistical KL Divergence Trajectory & p-value Boundary                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Root Cause Attribution Panel:                                                          │
│  [ Tab: SHAP Feature Attributions ]   |   [ Tab: Per-Feature KS Statistical Shift ]    │
│   • Sensor LIT101: 34.2% Contribution         • Sensor LIT101: p = 0.0001 (Drifted)   │
│   • Sensor DPIT301: 22.8% Contribution        • Sensor DPIT301: p = 0.0004 (Drifted)  │
│   • Sensor P402: 18.5% Contribution           • Sensor P402: p = 0.0012 (Drifted)     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ Operations & Control Bar:                                                              │
│  [ Simulate Drift ]   [ Trigger Retrain ]   [ Reset Dashboard ]   [ Admin Config ]     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### UI Features & Components
1. **Header Bar**: Displays real-time API connectivity status, setup progress indicator, active model version tag (`v1`, `v2`, etc.), overall system state (`Healthy`, `Warning`, `Critical`), and dynamic alert badges.
2. **Executive KPI Cards**:
   - **Model Health Score Card**: Interactive progress gauge displaying MHS value and status breakdown.
   - **Drift Status Card**: Displays composite drift severity, KL $p$-value, and classified drift type (`gradual`, `sudden`, etc.).
   - **Epistemic Uncertainty Card**: Shows mean model variance score from the RNN ensemble.
   - **Model Version & Performance Card**: Displays accuracy/F1-score, ground truth label availability, and active version ID.
3. **Live Telemetry SVG Charts**:
   - **Reconstruction Error Chart**: Real-time SVG line chart plotting batch reconstruction MSE against the dynamic threshold line over time, complete with hover tooltips and drift point markers.
   - **Statistical KL Divergence Chart**: Tracks KL divergence trajectory alongside the critical $p=0.01$ threshold.
4. **Root Cause Attribution Tabs**:
   - **SHAP Tab**: Visual bar chart ranking top drifted features by contribution to reconstruction loss.
   - **KS Test Tab**: Tabular view displaying Kolmogorov-Smirnov statistics, $p$-values, and drift confirmation flags per sensor feature.
5. **Control Panel & Admin Modals**:
   - **Simulate Drift Button**: Triggers simulated IoT stream drift injection from `current_drift` dataset.
   - **Trigger Retrain Button**: Manually invokes fine-tuning with SMOTE and validation gate checking.
   - **Reset Dashboard Button**: Clears buffer and resets monitoring history for clean demo runs.
   - **Admin Configuration Modal**: Live modal allowing operators to tune window size, drift thresholds, alpha levels, SMOTE toggles, active classifier type, and validation gate rules.

---

## 5. From Start to Finish: End-to-End System Lifecycle

### A. Initialization Phase
1. **Server Boot**: FastAPI server starts up.
2. **Reference Scaling**: Loads baseline dataset (`reference` window of `synthetic_swat.csv`) and fits `MinMaxScaler`.
3. **Model Setup**: Trains `ProductionClassifier` (Random Forest by default) on reference features.
4. **Background Component Setup**:
   - Trains Variational Autoencoder (VAE) for 10 epochs on scaled reference data.
   - Trains 3-member LSTM RNN Ensemble for 5 epochs for uncertainty tracking.
   - Setup progress transitions: `pending` $\rightarrow$ `training_ae` $\rightarrow$ `training_rnn` $\rightarrow$ `ready`.

### B. Prediction and Ingestion Path
1. **Prediction Requests**: Client or simulator sends feature payload (51 sensor values) to `POST /predict`.
2. **Real-time Inference**: `ProductionClassifier` computes instant prediction and confidence score, returning `PredictResponse`.
3. **Buffering**: Payload is appended to an in-memory queue asynchronously.
4. **Flush Trigger**: When buffer reaches window size (500 samples), contents are cloned, queue is flushed, and `RadarOrchestrator.run_monitoring_cycle()` launches in a background thread.

### C. The 7-Layer Monitoring Pipeline
1. **Layer 1: Ingestion & Preprocessing**: Min-Max scales batch features using reference parameters.
2. **Layer 2: Pure Distribution Drift Detection**: Passes batch through VAE, computes MSE reconstruction errors, calibrates via Isotonic Regression, runs 1,000-shuffle Permutation Test for KL divergence $p$-value, calculates Wasserstein distance, and runs per-feature KS tests.
3. **Layer 3: Epistemic Uncertainty Estimation**: RNN Ensemble runs 10 Monte Carlo Dropout forward passes to estimate model variance.
4. **Layer 4: Drift Signal Fusion & Categorization**: Fuses statistical metrics into composite drift score and classifies drift type.
5. **Layer 5: Model Health Score (MHS) Calculation**: Aggregates accuracy, drift, uncertainty, and stability into a unified MHS score and maps state (`Healthy`, `Warning`, `Critical`).
6. **Layer 6: Root Cause Attribution & Retraining**:
   - Runs SHAP (KernelExplainer) if drift is confirmed to identify root-cause features.
   - If MHS is `Critical`, applies Borderline-SMOTE oversampling and triggers VAE/Classifier fine-tuning.
7. **Layer 7: Validation Gate & Alerting**:
   - Evaluates retrained model candidate via Validation Gate. If passed, updates model version in SQLite registry (`v1` $\rightarrow$ `v2`) and promotes candidate.
   - Formats alert payload and appends result to monitoring history.

---

## 6. The Dataset & Simulation Mechanics

### The Dataset (`synthetic_swat.csv`)
Based on the Secure Water Treatment (SWaT) industrial IoT architecture:
- **51 Continuous Features**: Sensor measurements including Flow Interface Transmitters (`FIT`), Level Indicators (`LIT`), Analytical Indicators (`AIT`), Pressure Indicators (`PIT`), Differential Pressure (`DPIT`), and pump/valve actuators.
- **Labels**: Binary indicator (`0` = nominal operation, `1` = cyber-attack or mechanical anomaly).

Divided into 3 chronological windows:
1. **`reference`** (20,000 samples): Nominal baseline data used for training baseline scalers, VAE, RNN, and initial classifier.
2. **`current_stable`** (10,000 samples): Nominal production stream keeping monitoring metrics in `Healthy` state.
3. **`current_drift`** (8,000 samples): Injects intentional drift into sensors **LIT101**, **DPIT301**, and **P402** (ramp drift, noise, and frequency shifts), triggering `Warning` and `Critical` alerts.

---

## 7. Transitioning from Demo to Production Architecture

To transition this architecture into an enterprise production environment monitoring live models:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PRODUCTION ARCHITECTURE                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘

    Incoming Requests                                    Log Capture Middleware
  ──────────────────────► [ Production API Server ] ─────────────────────────┐
                                (FastAPI, Triton, TorchServe)                 │
                                                                              ▼
  ┌──────────────────────────────────────────────────────────────────────────┴───────────┐
  │                           Async Streaming Ingestion Broker                           │
  │                      (Apache Kafka, AWS Kinesis, RabbitMQ)                           │
  └────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                   │ Raw Payloads (Features + Preds)
                                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                            Stream Processing / Batching                              │
  │                      (Apache Spark, Flink, Celery Workers)                           │
  └────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                   │ Evaluated Batches
                                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                            Drift Detection & Analysis Engine                         │
  │                      (Alibi Detect, Evidently AI, Great Expectations)                │
  └────────────────────────┬────────────────────────────────┬────────────────────────────┘
                           │ Write Telemetry                │ Trigger Alerts
                           ▼                                ▼
  ┌────────────────────────┴──────────────┐        ┌────────┴────────────────────────────┐
  │          Databases & Storage          │        │       Alerting & Orchestration      │
  │  • TimescaleDB / InfluxDB (Metrics)   │        │  • PagerDuty / Slack / Email        │
  │  • PostgreSQL (Metadata & Versioning) │        │  • Airflow / Kubeflow (Retraining)  │
  │  • S3 / GCS (Model Artifact Store)    │        │                                     │
  └────────────────────────┬──────────────┘        └─────────────────────────────────────┘
                           │ Poll Telemetry
                           ▼
  ┌────────────────────────┴──────────────┐
  │          Observability Dashboard      │
  │  • Next.js Enterprise Admin Console   │
  │  • Grafana Monitoring Dashboards      │
  └───────────────────────────────────────┘
```

### Production Migration Highlights
1. **Streaming Ingestion**: Replace in-memory buffer with **Apache Kafka** or **AWS Kinesis** logging middleware for fault-tolerant inference logging.
2. **Distributed Stream Processing**: Use **Apache Spark** or **Flink** windowing to compute drift over massive request streams asynchronously.
3. **Time-Series Persistence**: Store reconstruction errors and metrics in **TimescaleDB** or **Prometheus**, and persist model artifacts to **S3 / MLflow**.
4. **Delayed Ground-Truth Processing**: Join incoming delayed ground-truth labels by `prediction_id` to evaluate actual real-world precision/recall decay over time.
5. **Automated MLOps Pipelines**: Connect critical alerts to **Airflow** or **Kubeflow** pipelines for automated retraining and validation.
