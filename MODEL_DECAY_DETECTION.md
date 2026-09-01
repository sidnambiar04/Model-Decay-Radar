# Model Decay Radar: Architecture, Algorithms, and Production Transition Guide

This document provides a detailed, step-by-step breakdown of how the Model Decay Radar system works, what algorithms it uses, what it simulates, and how you can adapt this architecture into a production-grade system for monitoring models already deployed in enterprise environments.

---

## 1. What the Project Does (High-Level Overview)

The Model Decay Radar is an end-to-end monitoring MVP designed to solve a critical ML problem: **silent model degradation (data drift)**. 

When ML models are deployed in production, the data they receive can slowly drift away from the distribution of the original training data (due to sensor wear, behavioral shifts, or environmental changes). Because the model continues to output predictions, standard health checks (such as checking if the API is "up") will not catch this degradation. 

This project acts as an automated "radar" that sits in the background of a prediction API. It ingests incoming inputs, analyzes them in chunks, assesses statistical distribution shifts and uncertainty levels, explains *why* the data shifted (attributing blame to specific features), triggers alerts or fine-tuning, and displays this live telemetry on a Next.js dashboard.

---

## 2. From Start to Finish: End-to-End System Lifecycle

Here is exactly what happens when you start the project and trigger the simulator:

### A. Initialization Phase
1. **Server Boot**: The FastAPI server starts.
2. **Reference Scaling**: The server loads a historical baseline dataset (the `reference` window of the synthetic CSV) and fits a `MinMaxScaler`.
3. **Model Setup**: A `ProductionClassifier` (a Random Forest model) is trained on this reference dataset.
4. **Background Setup Thread**: 
   - A background thread starts to train the statistical monitoring models on the scaled reference dataset.
   - It trains a **Variational Autoencoder (VAE)** for 10 epochs (used for reconstruction-based drift).
   - It trains a **3-member RNN (LSTM) Ensemble** for 5 epochs (used for uncertainty tracking).
   - The setup progress goes from `pending` -> `training_ae` -> `training_rnn` -> `ready`.

### B. Prediction and Ingestion Path
1. **Prediction Requests**: An external client or simulated process sends a payload containing 51 sensor features to `POST /predict`.
2. **Real-time Prediction**: The server scales the features, uses the `ProductionClassifier` to immediately output a class prediction and confidence score, and returns the response to the caller.
3. **Buffering**: In the background, the server adds the input features and labels to an in-memory buffer list.
4. **Flush Trigger**: Once the buffer fills up to a set window size (500 samples), the server clones the buffer contents, flushes the active queue, and starts a **background monitoring cycle thread** so that the `/predict` endpoint remains fast and unblocked.

### C. The Monitoring Cycle (7-Layer Pipeline)
The background orchestrator executes the following layers sequentially on the 500-sample batch:
1. **Ingestion & Scaling**: The raw batch is min-max scaled using the scaling factors computed from the historical reference data.
2. **Imbalance Handling**: If labels are provided and class distribution is heavily skewed, **Borderline SMOTE** is applied to oversample the minority class.
3. **Drift Detection (Autoencoder + Permutation Test)**:
   - The batch is passed through the pre-trained Autoencoder. 
   - Reconstruction errors (how poorly the autoencoder reconstructs the inputs) are calculated.
   - Calibration mappings (isotonic regression fit on reference data) map these reconstruction errors to a [0, 1] scale.
   - A **Permutation Test** is run: it mixes the reference reconstruction scores and the current batch scores, repeating this 1000 times, to compute the statistical **KL Divergence** and an empirical **p-value**.
   - If the `p-value < 0.01`, **data drift is officially confirmed**.
4. **Uncertainty Estimation**: The RNN ensemble computes forward passes with **Monte Carlo (MC) Dropout** enabled, estimating epistemic uncertainty (lack of familiarity with the new data pattern).
5. **Model Health Score (MHS) Fusion**:
   - The system aggregates indicators into a single score: `MHS = 0.35 * accuracy + 0.25 * (1 - drift) + 0.20 * (1 - uncertainty) + 0.20 * stability`.
   - Health is mapped to status states: `Healthy` (> 0.85), `Warning` (0.65 - 0.85), or `Critical` (< 0.65).
6. **SHAP Feature Attribution (Explanations)**:
   - If drift is confirmed, a **SHAP KernelExplainer** runs on the Autoencoder reconstruction loss. It pinpoints exactly which of the 51 features contributed most to the high reconstruction error (revealing the root causes of the drift).
7. **Adaptive Fine-tuning (Retraining)**:
   - If the Model Health Score drops to `Critical`, a selective training loop is triggered in the background. The Autoencoder is fine-tuned for 5 epochs on the drifted batch to adapt its reconstruction thresholds to the new distribution.
8. **Alert Generation**:
   - The cycle compiles a `MonitoringResult` payload, logs it, and pushes it to an in-memory history array.
   - An alert message is formatted with details like: the p-value, KL divergence value, top 3 drifted sensors, and whether retraining was triggered.

### D. Visualization Path
1. **Next.js Polling**: The Next.js dashboard polls the `/monitoring/history` and `/health` endpoints every 3 seconds.
2. **Live Feed Rendering**: The UI displays current gauge metrics, active alert banners, lists of SHAP-attributed drifted features, and real-time custom SVG charts showing reconstruction error and KL divergence over time.

---

## 3. The Dataset & What Simulates What

### What are the "Samples"?
The samples are rows of data from an industrial IoT system. It mimics the **SWaT (Secure Water Treatment)** dataset, representing:
- **51 features**: Continuous physical measurements from water processing equipment, including:
  - **FIT** (Flow Interface Transmitter)
  - **LIT** (Level Indicator Transmitter)
  - **AIT** (Analytical Indicator Transmitter)
  - **PIT** (Pressure Indicator Transmitter)
  - **DPIT** (Differential Pressure Transmitter)
  - Actuator signals like valve status and pump feedback.
- **Labels**: Binary values where `0` represents nominal behavior and `1` represents an operational anomaly or cyber-attack.

The dataset is pre-divided into three chronological windows to facilitate testing:
1. **`reference`** (20,000 samples): Clean, nominal behavior used for training scaled parameters, classifiers, VAE, and RNN.
2. **`current_stable`** (10,000 samples): Production data that follows the original distribution. Monitoring scores remain green/healthy.
3. **`current_drift`** (8,000 samples): Injects intentional drift into three specific sensors: **LIT101**, **DPIT301**, and **P402** (introducing gradual ramps, frequency changes, and noise). Ingesting this window triggers the warning and critical radar thresholds.

### What is Real vs. Simulated?

To keep the MVP lightweight and executable on a local workstation, certain elements are simulated:
- **Data Ingestion Stream**: Rather than reading from a live event bus (like Apache Kafka), the data stream is simulated by reading chunks of the pre-generated synthetic SWaT CSV. Clicking "Simulate Drift" instructs the backend API to inject blocks of these stable or drifted CSV rows into the prediction pipeline.
- **Production ML Model**: The prediction API targets a fast RandomForest classifier trained at startup. In production, this would be your complex deep learning or gradient-boosted model.
- **MHS Accuracy**: In a real system, you might not receive true labels immediately to compute exact accuracy. For this demo, the accuracy parameter within the Model Health Score is mathematically simulated (starting at 0.92, dropping on drift, and adjusting slightly during stability phases).
- **Telemetry Storage**: Telemetry logs and batch results are saved in an in-memory queue rather than a database.

---

## 4. Algorithms Used & Their Purpose

| Algorithm | How it is used | Purpose |
| :--- | :--- | :--- |
| **Variational Autoencoder (VAE)** | Compresses 51-dimensional inputs into a lower-dimensional bottleneck, then reconstructs them. | **Drift Detection**. When incoming data drifts from training data, the VAE fails to reconstruct it well, resulting in high reconstruction error. |
| **Isotonic Regression** | Fits a monotonic calibration mapping on reference reconstruction scores. | **Score Normalization**. Maps raw, arbitrary reconstruction MSE values into a normalized 0-to-1 probability range. |
| **KL Divergence Permutation Test** | Calculates KL divergence between reference and current batch scores, shuffling labels 1000 times to compute a p-value. | **Statistical Significance**. Proves whether the observed score shifts are statistically significant or just random noise (drift confirmed if $p < 0.01$). |
| **Borderline SMOTE** | Generates synthetic minority class samples along decision borders. | **Class Imbalance**. Oversamples rare anomalies to prevent evaluation metrics from being biased by highly imbalanced data. |
| **RNN (LSTM) Ensemble + MC Dropout** | Combines 3 LSTM networks trained on time-series reconstruction errors, running 10 stochastic forward passes with dropout active. | **Epistemic Uncertainty**. Quantifies how "unfamiliar" the current time-series pattern is to the models, separating simple outliers from core systematic distribution changes. |
| **SHAP (KernelExplainer)** | Approximates Shapley values by perturbing sensor features and measuring their effect on the VAE reconstruction error. | **Root-Cause Attribution**. Explains *which* specific sensors are responsible for the drift, letting operators know what hardware or processes are failing. |
| **Selective Fine-Tuning** | Re-runs backpropagation for 5 epochs using the newly drifted batch on the VAE. | **Adaptive Learning**. Temporarily adjusts the autoencoder to the shifted baseline so that it doesn't alert indefinitely after a known process change. |

---

## 5. Transitioning from a Demo to a Production Product

To turn this MVP into an enterprise-grade platform that monitors already deployed production models, you must replace the simulated and local-in-memory constructs with scalable, production-ready components.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               PRODUCTION ARCHITECTURE                                  │
└────────────────────────────────────────────────────────────────────────────────────────┘

    Incoming Requests                                    Log Capture Middleware
  ──────────────────────► [ Production API Server ] ─────────────────────────┐
                                (Flask, FastAPI, Triton, etc.)               │
                                                                             ▼
  ┌──────────────────────────────────────────────────────────────────────────┴───────────┐
  │                           Async Streaming Ingestion Broker                           │
  │                      (Apache Kafka, RabbitMQ, or AWS Kinesis)                │
  └────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                   │ Raw Payloads (Features + Preds)
                                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                            Stream Processing / Batching                              │
  │                      (Apache Spark, Flink, or Celery Workers)                        │
  └────────────────────────────────────────────────┬─────────────────────────────────────┘
                                                   │ Evaluated Batches
                                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────┐
  │                            Drift Detection & Analysis Engine                         │
  │                      (Alibi Detect, Evidently AI, or Great Expectations)             │
  └────────────────────────┬────────────────────────────────┬────────────────────────────┘
                           │ Write Telemetry                │ Trigger Alerts
                           ▼                                ▼
  ┌────────────────────────┴──────────────┐        ┌────────┴────────────────────────────┐
  │          Databases & Storage          │        │       Alerting & Orchestration      │
  │  • TimescaleDB / InfluxDB (Metrics)   │        │  • PagerDuty / Slack / Email        │
  │  • PostgreSQL (Alert Log History)     │        │  • Airflow / Kubeflow (Retraining)  │
  │  • S3 / GCS (Model Artifact Registry) │        │                                     │
  └────────────────────────┬──────────────┘        └─────────────────────────────────────┘
                           │ Poll Logs
                           ▼
  ┌────────────────────────┴──────────────┐
  │          Visualization UI             │
  │  • Grafana Dashboards                 │
  │  • Custom Enterprise Next.js Admin    │
  └───────────────────────────────────────┘
```

Here are the key architectural migrations required for production:

### A. Logging & Data Ingestion (Replacing the In-Memory Buffer)
*   **The Demo**: Appends requests directly to a local list (`buffer_features`) in Python memory.
*   **Production**: 
    - Implement logging middleware on your deployed API gateways (e.g., FastAPI, Flask, Triton Inference Server, or TorchServe) that captures inference payloads (inputs and outputs) asynchronously.
    - Write these payloads to a message broker such as **Apache Kafka**, **AWS Kinesis**, or **RabbitMQ**. This guarantees that if the monitoring service crashes, no inference logs are lost, and the prediction path remains fast and isolated.

### B. Scalable Data Processing (Replacing the Background Thread)
*   **The Demo**: Uses Python's standard `threading.Thread` to execute the monitoring cycle.
*   **Production**:
    - Use a distributed processing engine like **Apache Spark**, **Apache Flink**, or a task worker system like **Celery** to consume batches from Kafka.
    - Spark/Flink can window incoming data by time (e.g., hourly, daily) or by count (e.g., every 10,000 requests) to run drift calculations over large datasets without consuming API server RAM.

### C. Persistent Storage (Replacing the In-Memory History List)
*   **The Demo**: Appends outputs to `monitoring_history = []`. If the server restarts, all historical telemetry is lost.
*   **Production**:
    - Save inference payloads and metrics (reconstruction errors, KL divergence scores, and predictions) to a time-series database like **TimescaleDB**, **InfluxDB**, or **Prometheus**.
    - Save alerts, training cycles, metadata, and SHAP attributions in a relational database like **PostgreSQL**.
    - Store trained model checkpoints (reference scalers, Autoencoders, and active production weights) in a secure artifact store like **AWS S3**, **Google Cloud Storage (GCS)**, or an **MLflow Model Registry**.

### D. Production-Grade Drift Engines (Replacing Custom Scratch Code)
*   **The Demo**: Uses scratch PyTorch scripts for the VAE, RNN, and permutation tests.
*   **Production**:
    - Leverage robust, vetted open-source monitoring libraries:
      - **Evidently AI**: Outstanding for tabular and text data drift, generating reports, and calculating metrics like Kolmogorov-Smirnov (KS) tests, Wasserstein Distance, or Population Stability Index (PSI).
      - **Alibi Detect**: Excellent for outlier detection, concept drift (using Maximum Mean Divestment - MMD, or Chi-Square tests), and adversarial drift.
      - **Great Expectations**: Perfect for verifying schema adherence and data quality constraints before running ML pipelines.

### E. Feedback Loops & Delayed Labels (Solving the Ground-Truth Problem)
*   **The Demo**: Assumes class accuracy is simulated because real-world labels are usually delayed.
*   **Production**:
    - Build a **delayed feedback handler**. Real-world labels (e.g., whether a transaction was actually fraudulent, or if a valve actually failed) often arrive days or weeks later. 
    - Store inference predictions with a unique `prediction_id` in database logs. When labels arrive from the system or human labelers, join them using the `prediction_id` to compute actual, non-simulated performance decay (F1-score, Precision, Recall).

### F. Enterprise Alerting & Orchestration (Replacing Console Messages)
*   **The Demo**: Formats a text message displayed on a dashboard card.
*   **Production**:
    - Integrate alerting with standard operations dashboards. When p-values indicate statistically significant drift or the Model Health Score drops below a critical threshold, trigger webhooks to **Slack**, **Microsoft Teams**, **PagerDuty**, or send SMS/emails via **Twilio** / **SendGrid**.
    - Link critical alerts to workflow orchestrators (like **Apache Airflow**, **Prefect**, or **Kubeflow Pipelines**) to automatically run automated model retraining pipelines on fresh data, run validation tests, and register the updated weights to the registry.
