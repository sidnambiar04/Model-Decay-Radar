# Model Decay Radar

Real-time ML model monitoring demo inspired by industrial IoT (SWaT-style water treatment sensors). Simulates a deployed model receiving live sensor data, detects when the data distribution drifts, explains which sensors changed, and triggers alerts/retraining — all visible on a live dashboard.

This is **not** a full production MLOps platform. It is an MVP that wires together drift detection, uncertainty, SHAP, and alerting into one runnable demo.

---

## Recent Updates & Key Changes

1. **Dashboard Migration (Streamlit to Next.js)**: The original Python/Streamlit dashboard (`dashboard/app.py` on port 8501) has been migrated to a modern **Next.js** web application in `frontend/` running on `http://localhost:3000`.
2. **Interactive UI with Custom SVG Charts**: The frontend dashboard now features live-updating custom SVG charts for Reconstruction Error, KL Divergence, and Model Health Score, removing the need for heavy external visualization plotting libraries.
3. **CORS Enabled**: Configured CORSMiddleware in FastAPI (`api/server.py`) to permit requests from the Next.js development origin (`http://localhost:3000`).
4. **Improved Server Automation (`run.sh` / `stop.sh`)**: Updated the startup and termination shell scripts to seamlessly launch and terminate both the FastAPI backend and Next.js frontend concurrently.

---

## High-Level Idea

Imagine you deployed an ML model on 51 industrial sensors (flow, level, pressure, actuators, etc.). Over time, the plant behavior changes — sensors drift, valves wear, processes shift. Your model may silently get worse.

**Model Decay Radar** watches incoming data in the background and answers:

1. Is the model still healthy?
2. Has data drift occurred (statistically)?
3. Which sensors caused the drift?
4. Should we alert or retrain?

The user/IoT device only calls **`POST /predict`** and gets a prediction back. Monitoring happens invisibly in the background.

---

## Major Project System Architecture

```
┌─────────────┐     POST /predict      ┌─────────────────────────────────────────┐
│ Client/IoT  │ ─────────────────────► │ FastAPI Server (port 8000)              │
│ Simulation  │                        │  • Active Production Classifier (RF/GB) │
│ Engine      │                        │  • Raw Sample Buffer (500 samples)      │
└─────────────┘                        │  • Delayed Ground Truth Tracker         │
                                       └────────────────────┬────────────────────┘
                                                            │ every 500 samples
                                                            ▼
                                       ┌─────────────────────────────────────────┐
                                       │ RadarOrchestrator Pipeline (7 Layers)   │
                                       │  1. Centralized Config (RadarConfig)     │
                                       │  2. Pure Raw Distribution Drift (VAE)   │
                                       │  3. Multi-Detector Fusion Engine        │
                                       │  4. MC Dropout Uncertainty (T=50)       │
                                       │  5. Dynamic Reweighted MHS               │
                                       │  6. Side-by-Side Root Cause (SHAP vs KS) │
                                       │  7. Candidate Retraining & SMOTE        │
                                       └────────────────────┬────────────────────┘
                                                            │
                                                            ▼
                                       ┌─────────────────────────────────────────┐
                                       │ Validation Gate & SQLite Model Registry │
                                       │  • Compares Candidate vs Active Model   │
                                       │  • Version Tracking (v1, v2, v3)         │
                                       │  • Post-Promotion Reference Update Gate │
                                       └────────────────────┬────────────────────┘
                                                            │ MonitoringResult
                                                            ▼
┌─────────────┐     polls every 3s     ┌─────────────────────────────────────────┐
│ Next.js     │ ◄───────────────────── │ In-memory Monitoring History Log        │
│ Dashboard   │   /monitoring/*        │ (last 200 results)                      │
│ (port 3000) │                        └─────────────────────────────────────────┘
└─────────────┘
```

### Startup sequence (`run.sh`)

1. Generates `data/synthetic_swat.csv` if missing
2. Starts **FastAPI** on `http://localhost:8000`
3. Starts **Next.js frontend dashboard** on `http://localhost:3000`
4. On API startup, trains **Autoencoder (10 epochs)** + **RNN ensemble (5 epochs)** on reference data (~1 min)

---

## Quick Start & Running the Project

### Installation

1. **Python Dependencies** (Virtual environment recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. **Next.js Frontend Dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Services

#### Option A: One-Command Startup (Bash / Git Bash / Linux / macOS)
Simply run the startup script:
```bash
bash run.sh
```
Wait until `http://localhost:8000/health` shows `"orchestrator_ready": true` (takes ~1 minute for background training), then open:
- **Next.js Dashboard**: [http://localhost:3000](http://localhost:3000)
- **FastAPI Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

To stop the services:
```bash
bash stop.sh
```

#### Option B: Manual Startup (Windows Command Prompt or PowerShell)
If you don't have Bash / Git Bash, you can start the services manually in separate terminals:

1. **Start the FastAPI Backend**:
   In your first terminal (at the project root directory):
   - **PowerShell**:
     ```powershell
     $env:PYTHONPATH="src;pipeline;."
     python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --log-level info
     ```
   - **Command Prompt (CMD)**:
     ```cmd
     set PYTHONPATH=src;pipeline;.
     python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --log-level info
     ```

2. **Start the Next.js Frontend**:
   In your second terminal (navigate to the `frontend/` directory):
   ```bash
   cd frontend
   npm run dev
   ```

Once the backend log shows `Orchestrator ready`, open [http://localhost:3000](http://localhost:3000) and click **Simulate Drift** in the header to run the simulation.

---

## Folder Structure

| Path | Role |
|------|------|
| `run.sh` / `stop.sh` | Start/stop both backend and frontend services |
| `requirements.txt` | PyTorch, FastAPI, SHAP, imbalanced-learn, etc. |
| `data/synthetic_swat.csv` | Pre-generated demo dataset (38,000 rows) |
| `api/server.py` | FastAPI server — prediction endpoint, buffer, monitoring triggers |
| `frontend/` | Next.js TypeScript web application (modern UI dashboard) |
| `dashboard/` | Legacy Streamlit monitoring UI (archived) |
| `pipeline/orchestrator.py` | Core 7-layer monitoring pipeline |
| `src/autoencoder.py` | Variational autoencoder for drift detection |
| `src/calibration.py` | Score calibration + KL divergence permutation test |
| `src/uncertainty.py` | 3-member RNN ensemble with MC Dropout |
| `src/imbalance_handler.py` | Borderline SMOTE for class imbalance |
| `src/data_pipeline.py` | MinMax scaling from reference window |
| `src/generate_synthetic_swat.py` | Creates synthetic SWaT-like sensor data |
| `example_client.py` | Simulates an IoT device sending 50 `/predict` calls |
| `logs/` | API and frontend logs |

---

## The Synthetic Dataset

**File:** `data/synthetic_swat.csv`

| Property | Value |
|----------|-------|
| Rows | 38,000 |
| Features | 51 sensor/actuator columns (FIT, LIT, AIT, DPIT, PIT, MV, P, UV tags) |
| Label | `0` = normal, `1` = anomaly (640 anomalies, ~1.7%) |
| `_window` column | Splits data into 3 time phases |

### Three time windows

| Window | Rows | Meaning |
|--------|------|---------|
| `reference` | 20,000 | Historical baseline — used to train AE/RNN and fit scaler |
| `current_stable` | 10,000 | Recent production data, still similar to reference |
| `current_drift` | 8,000 | Drift injected into 3 sensors: **LIT101**, **DPIT301**, **P402** (ramp + frequency change + noise) |

The demo is designed so that monitoring cycles on **stable** data look healthy, then degrade when **drift** data flows in.

Regenerate dataset:

```bash
python src/generate_synthetic_swat.py
```

---

## The 7-Layer Monitoring Pipeline

Implemented in `pipeline/orchestrator.py`. Runs every time **500 samples** fill the buffer (`WINDOW_SIZE = 500` in `api/server.py`).

### Layer 1 — Data ingestion (API server)

- Incoming features are scaled with a **MinMaxScaler** fit on the reference window (`src/data_pipeline.py`)
- Samples buffered until `WINDOW_SIZE = 500`
- When full → background monitoring cycle starts, buffer clears

### Layer 2 — Class imbalance (SMOTE)

- If labels are provided and minority/majority ratio < 0.2 → **BorderlineSMOTE** oversamples minority class
- Implemented in `src/imbalance_handler.py`

### Layer 3 — Drift detection (Autoencoder)

- **Variational Autoencoder** trained on reference data (`src/autoencoder.py`)
- Reconstruction error = `0.7 × MSE + 0.3 × KL`
- Errors calibrated to 0–1 via **Isotonic Regression** (`src/calibration.py`)
- **Permutation test** (1000 permutations): compares reference vs current score histograms via **KL divergence**
- Drift confirmed if **p-value < 0.01**
- Dynamic threshold = `mean(ref_errors) + std(ref_errors)`

### Layer 4 — Uncertainty (RNN Ensemble)

- 3 LSTM models trained on reference reconstruction-error time series (`src/uncertainty.py`)
- **MC Dropout** (10 forward passes) estimates epistemic uncertainty
- Normalized against reference uncertainty → 0–1 score

### Layer 5 — Model Health Score (MHS)

Weighted fusion in `pipeline/orchestrator.py`:

```
MHS = 0.35×accuracy + 0.25×(1-drift) + 0.20×(1-uncertainty) + 0.20×stability
```

| MHS | Status |
|-----|--------|
| > 0.85 | Healthy |
| 0.65 – 0.85 | Warning |
| < 0.65 | Critical |

**Note:** `accuracy` is **simulated** (starts at 0.92, drops on drift, rises slightly when stable) — not computed from real model predictions vs labels.

### Layer 6 — Interpretation & retraining

- **SHAP** (KernelExplainer on AE loss) runs **only when drift is confirmed** → top 10 drift-causing sensors
- **Selective retraining** runs **only when MHS is Critical** → fine-tunes autoencoder 5 epochs on drift batch, updates reference errors/threshold

### Layer 7 — Alerting

- Alert levels: `none` / `warning` / `critical`
- Human-readable messages with p-value, KL, top features, and retraining status

Each cycle produces a `MonitoringResult` dataclass stored in memory (last 200 results).

### MonitoringResult fields

| Field | Description |
|-------|-------------|
| `batch_id` | Incrementing cycle number |
| `timestamp` | Unix timestamp |
| `mean_reconstruction_error` | Mean AE loss on batch |
| `dynamic_threshold` | Reference-based error threshold |
| `observed_kl` | KL divergence (reference vs current scores) |
| `p_value` | Permutation test p-value |
| `drift_confirmed` | Boolean, p < 0.01 |
| `drift_severity` | 0–1 normalized severity |
| `mean_uncertainty` | Normalized MC Dropout uncertainty |
| `mhs` | Model Health Score 0–1 |
| `mhs_status` | Healthy / Warning / Critical |
| `top_drift_features` | `[{feature, importance}]` from SHAP |
| `smote_applied` | Whether SMOTE ran |
| `samples_before_smote` / `samples_after_smote` | Batch sizes |
| `alert_level` | none / warning / critical |
| `alert_message` | Human-readable alert text |
| `retraining_triggered` | Whether AE fine-tune ran |

---

## The ML "Model" (Important Limitation)

The deployed model in `api/server.py` is a **`DummyMLModel`**:

```python
prediction = 1.0 if mean(features) > 0.5 else 0.0
confidence = based on distance from 0.5
```

It is a placeholder. In production you would swap this for a real sklearn/PyTorch model. The monitoring pipeline does **not** use this dummy model's predictions for drift — it uses the autoencoder on raw features.

---

## FastAPI — All Endpoints

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### `GET /health`

Server + setup status.

Returns:

- `status` — `"ok"`
- `orchestrator_ready` — AE/RNN training finished?
- `buffer_size` / `window_size` (500)
- `monitoring_cycles_completed`

### `POST /predict` — Main production endpoint

**Request:**

```json
{
  "features": {"LIT101": 45.2, "FIT101": 12.1},
  "label": 0
}
```

- `features`: dict of `{sensor_name: value}` for all 51 feature columns
- `label`: optional ground truth (may arrive late in real systems)

**Response:**

```json
{
  "prediction": 0.0,
  "confidence": 0.87,
  "model_status": "Healthy",
  "batch_buffered": 342
}
```

**Behavior:**

1. Validates feature dict against all 51 sensor columns
2. Scales features using reference-window scaler
3. Returns dummy prediction immediately (fast)
4. Buffers sample
5. At 500 samples → triggers background monitoring cycle

### `GET /monitoring/latest`

Most recent `MonitoringResult` — used by dashboard for SHAP, alerts, etc.

Returns `{"status": "no_data", ...}` if no cycles completed yet.

### `GET /monitoring/history?limit=50`

Last N monitoring cycles — powers all time-series charts.

Returns: `{"results": [...], "total": N}`

### `GET /monitoring/status`

Quick summary for dashboard header:

- `mhs`, `status`, `alert_level`, `drift_confirmed`, `cycles`

### `POST /admin/reset`

Clears buffer and monitoring history (demo restart).

### `GET /admin/simulate?n_stable=1000&n_drift=1500`

**Demo shortcut** — injects data from `synthetic_swat.csv` without a real IoT client:

- `n_stable`: samples from `current_stable` window
- `n_drift`: samples from `current_drift` window
- Triggers monitoring cycles in background (~5 cycles for default 1000+1500 samples)

This is what the dashboard **"Inject Simulation Data"** button calls.

---

## Next.js Frontend Dashboard

**URL:** [http://localhost:3000](http://localhost:3000)  
**Directory:** `frontend/`  
A modern web application built using Next.js (React 19, TypeScript) and styled with Tailwind CSS. It polls the FastAPI backend every **3 seconds** using a custom hook (`useMonitoring`) to render real-time visualizations.

### Header Controls & Status

The header provides live indicators and controls for running the simulator:
- **Connection Status**: Green dot showing if the client can reach the FastAPI backend.
- **Orchestrator Setup Status**: Displays setup progress (`pending`, `training_ae`, `training_rnn`, `ready`, or `failed`).
- **Simulate Drift Button**: Triggers the `/admin/simulate` endpoint to inject stable and drifted samples (only active when orchestrator status is `ready`).
- **Reset All Button**: Clears all monitoring history and the prediction buffer to start a clean session.
- **Buffer Progress**: Visual progress bar showing how close the current incoming prediction stream is to triggering the next monitoring cycle (500 samples).
- **Cycle Count**: Total completed evaluation cycles.

### Dashboard Layout & Sections

#### 1. Metric Indicator Grid
Four primary cards indicating system health:
- **Model Health Score (MHS)**: 0-100% score mapped to dynamic colors (`Healthy` in green, `Warning` in amber, `Critical` in rose).
- **Drift Detected**: Clearly states `YES` (with flashing indicator) or `NO` based on statistical p-value threshold (p < 0.01).
- **Alert Status**: Shows `Nominal` (green), `Warning` (amber), or `Critical` (rose) alerts.
- **KL Divergence**: Quantifies current batch distribution shift relative to the reference baseline.

#### 2. Alert Banner & Details
- **Latest Alert Box**: Shows a prominent, color-coded alert message describing the drift parameters, statistical indicators, and details.
- **Details Grid**: Shows the latest batch statistical indicators including p-value, SMOTE balance applications, and whether auto-encoder retraining occurred.

#### 3. Real-time SVG Charts
Unlike traditional dashboards, the Next.js app renders high-performance interactive SVGs for continuous metrics tracking:
- **Reconstruction Error vs. Threshold**: Tracks the mean reconstruction error against a dynamic, adaptive threshold. Drift points are marked with distinct icons.
- **KL Divergence Trend**: Traces data distribution shifts over evaluation batches.
- **Model Health Score Trend**: Displays MHS over time, highlighting warning and critical bands.

#### 4. Diagnostic & Interpretability Tools
- **SHAP Feature Importance**: Runs SHAP analysis on the autoencoder reconstruction loss to identify the top drift-causing sensors. Features are listed in a table with a horizontal contribution bar.
- **Epistemic Uncertainty**: Shows the RNN ensemble uncertainty trend (MC Dropout) to differentiate data drift from out-of-distribution noise.

#### 5. Alert & Cycle History Log
A complete scrollable log table recording details of all historical cycles (batch ID, status, alert level, p-value, whether SMOTE or retraining ran, and the full text message).

---

## Example Client

**File:** `example_client.py`

Simulates a real IoT gateway:

- Reads 50 rows from `current_drift` window
- Sends each as `POST /predict` with features + label
- Prints prediction, confidence, model status, buffer progress
- 50ms delay between requests

```bash
python example_client.py
```

Use this instead of the simulate button if you want to see the buffer fill gradually (~10 runs of 50 samples to trigger one 500-sample cycle).

---

## Typical Demo Workflow

1. Run `bash run.sh`
2. Wait until `http://localhost:8000/health` shows `"orchestrator_ready": true` (~2–3 min)
3. Open dashboard at `http://localhost:8501`
4. Click **Inject Simulation Data**
5. Watch over ~15–30 seconds as monitoring cycles complete:
   - First cycles (stable data): MHS stays Healthy, low KL
   - Later cycles (drift data): KL rises, drift confirmed, MHS drops to Warning/Critical
   - SHAP highlights LIT101, DPIT301, P402
   - Critical state may trigger retraining markers on MHS chart

---

## What Is Real vs Simulated (MVP Gaps)

| Component | Status |
|-----------|--------|
| Autoencoder drift detection | Real (PyTorch, trained at startup) |
| Permutation test + KL divergence | Real |
| RNN uncertainty ensemble | Real |
| SHAP interpretation | Real (simplified background sampling) |
| SMOTE | Real |
| Selective AE retraining | Real (5-epoch fine-tune) |
| Deployed ML model | **Dummy placeholder** |
| Model accuracy in MHS | **Simulated**, not from real predictions |
| Data persistence | **In-memory only** (no DB/Redis) |
| Authentication | None |
| Real IoT streaming | Simulated via CSV injection |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI + Uvicorn |
| Frontend | Next.js (React 19 + TypeScript + Tailwind CSS) |
| ML | PyTorch (VAE + LSTM), scikit-learn (RandomForest), SHAP, imbalanced-learn |
| Data | Synthetic SWaT-like CSV with engineered drift |
| HTTP client | fetch API (Frontend) / httpx (Backend tests) |

---

## Key Configuration Constants

| Constant | Location | Default | Meaning |
|----------|----------|---------|---------|
| `WINDOW_SIZE` | `api/server.py` | 500 | Samples per monitoring cycle |
| `AE_EPOCHS` | `api/server.py` | 10 | Autoencoder training epochs at startup |
| `RNN_EPOCHS` | `api/server.py` | 5 | RNN ensemble training epochs at startup |
| `POLL_SEC` | `frontend/app/hooks/useMonitoring.ts` | 3 | Dashboard refresh interval (seconds) |
| `n_permutations` | `src/calibration.py` | 1000 | Permutation test iterations |
| Drift p-value threshold | `src/calibration.py` | 0.01 | Drift confirmed if p < this |
| MHS weights | `pipeline/orchestrator.py` | 0.35/0.25/0.20/0.20 | accuracy/drift/uncertainty/stability |

---

## Mental Model

Two parallel systems:

1. **Prediction path** (user-facing): fast `/predict` → RandomForest classifier → immediate response
2. **Monitoring path** (background): every 500 samples → full drift/uncertainty/SHAP/alert pipeline → dashboard

The dashboard is a **monitoring console**, not a model training UI. It shows whether your deployed model's input data is decaying over time.

---

## Dependencies

### Backend Dependencies (`requirements.txt`)
- `torch>=2.1.0`
- `scikit-learn>=1.3.0`
- `imbalanced-learn>=0.11.0`
- `shap>=0.44.0`
- `scipy>=1.11.0`
- `pandas>=2.0.0`
- `numpy>=1.24.0`
- `fastapi>=0.110.0`
- `uvicorn[standard]>=0.29.0`
- `httpx>=0.27.0`

### Frontend Dependencies (`frontend/package.json`)
- `next`: `16.2.12`
- `react`: `19.2.4`
- `react-dom`: `19.2.4`
- `lucide-react`: `^1.27.0`
- `tailwindcss`: `^4`
- `typescript`: `^5`
