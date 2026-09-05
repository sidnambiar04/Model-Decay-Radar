# Model Decay Radar Enhancement Plan

## 1. Executive Summary

**Model Decay Radar** is an intelligent Machine Learning Observability, Drift Detection, and Adaptive Model Lifecycle Management platform designed for mission-critical ML deployments. Deployed machine learning models operate in dynamic environments where real-world data distributions continuously evolve—a phenomenon known as **data drift** and **concept drift**. Standard system health metrics (such as uptime, latency, or HTTP 200 responses) fail to catch silent model degradation, leaving production systems vulnerable to inaccurate predictions and silent failures.

Model Decay Radar addresses this core MLOps challenge by implementing an end-to-end, multi-layered monitoring radar that sits alongside prediction APIs. It ingests production data streams asynchronously, calculates multi-detector statistical distribution shifts and epistemic uncertainties, attributes root-cause drift to individual features using game-theoretic SHAP and Kolmogorov-Smirnov tests, tracks an executive Model Health Score (MHS), and manages an adaptive retraining loop governed by a pre-deployment safety Validation Gate and a lightweight SQLite Model Registry.

This document serves as the master architectural blueprint and phased implementation roadmap for the Model Decay Radar major Computer Science project. It provides an accurate audit of the existing workspace, defines target enhancements across 13 safe phases, outlines two explicit demonstration modes (*Controlled Demonstration Mode* and *Real Data Monitoring Mode*), presents a 12-step presentation workflow, and establishes a 4-person engineering team collaboration plan.

---

## 2. Current Repository Audit

A comprehensive audit of the codebase reveals that the workspace possesses a mature, functional baseline spanning 16 verified pipeline stages tested in `tests/run_all_tests.py`. Below is the official classification of all major project features:

| Feature / Subsystem | Current Implementation Path | Current Status | Notes & Audit Observations |
| :--- | :--- | :--- | :--- |
| **Reference Baseline & Preprocessing** | `src/data_pipeline.py` | **WORKING** | Fits `MinMaxScaler` on historical reference window (`synthetic_swat.csv`). Normalizes 51 IoT continuous features cleanly. |
| **Prediction API & Asynchronous Buffer** | `api/server.py` | **WORKING & NEEDS IMPROVEMENT** | Non-blocking `/predict` endpoint buffers incoming samples. Triggers monitoring when queue hits 500 samples. Uses in-memory queues; needs persistent telemetry backing. |
| **Variational Autoencoder (VAE) Detector** | `src/autoencoder.py` | **WORKING** | PyTorch VAE tracking reconstruction MSE and KL loss penalty. Dynamically calculates reconstruction error thresholds. |
| **Isotonic Score Calibration** | `src/calibration.py` | **WORKING** | Monotonic regression mapping raw MSE errors to normalized [0, 1] drift scores. |
| **Permutation KL Divergence Test** | `src/calibration.py` | **WORKING** | 1,000-shuffle non-parametric permutation test yielding exact statistical $p$-values ($p < 0.01$ threshold). |
| **Distance & Univariate Metrics** | `src/calibration.py` | **WORKING** | Calculates Wasserstein Distance (Earth Mover's Distance) and per-feature 2-sample Kolmogorov-Smirnov (KS) tests. |
| **Streaming Drift Detectors** | `src/calibration.py`, `src/drift_fusion.py` | **PARTIALLY WORKING** | ADWIN and DDM integrated into signal fusion logic. Requires explicit label streaming connection for supervised signals. |
| **Epistemic Uncertainty Engine** | `src/uncertainty.py` | **WORKING** | 3-member LSTM RNN ensemble running 10 Monte Carlo Dropout forward passes per sample to estimate model variance. |
| **Class Imbalance Handler** | `src/imbalance_handler.py` | **WORKING** | Applies Borderline-SMOTE oversampling when minority class ratio falls below configurable threshold (`0.20`). |
| **Multi-Classifier Architecture** | `src/classifier.py` | **WORKING & NEEDS IMPROVEMENT** | Supports Random Forest, XGBoost / Gradient Boosting, and MLP. Handles 1-class baseline data via `DummyClassifier`. Needs unified serialization interface. |
| **Drift Signal Fusion Engine** | `src/drift_fusion.py` | **WORKING** | Fuses VAE MSE, KL $p$-value, KS ratio, Wasserstein distance, ADWIN, and DDM into composite score and drift categories. |
| **Model Health Score (MHS) Engine** | `src/drift_fusion.py`, `pipeline/orchestrator.py` | **WORKING** | Fuses Accuracy, Drift, Uncertainty, and Stability: $\text{MHS} = 0.35 \cdot \text{Acc} + 0.25 \cdot (1-\text{Drift}) + 0.20 \cdot (1-\text{Uncertainty}) + 0.20 \cdot \text{Stability}$. |
| **Side-by-Side Root Cause Attribution** | `src/root_cause.py` | **WORKING & NEEDS IMPROVEMENT** | Combines SHAP (KernelExplainer) feature attributions with per-feature KS test $p$-values. Needs performance optimization for fast rendering. |
| **Pre-Deployment Validation Gate** | `src/validation_gate.py` | **WORKING** | Compares retrained candidate models against active models on validation split before promotion (`f1_margin=0.02`, `min_acc=0.80`). |
| **Versioned Model Registry** | `src/model_registry.py` | **WORKING & PARTIALLY WORKING** | SQLite DB (`models/model_registry.db`) tracking model version tags (`v1`, `v2`), hyperparams, metrics, and promotion status. Needs binary blob storage linking. |
| **Scenario Simulator Engine** | `src/simulator.py` | **WORKING & SIMULATED** | Generates feature batches for `normal`, `gradual`, `sudden`, `imbalance`, and `recovery` scenarios using `synthetic_swat.csv`. Modifies inputs only. |
| **Two Explicit Operating Modes** | `api/server.py`, `frontend/app/page.tsx` | **PARTIALLY WORKING / NEEDS UI TOGGLE** | Backend supports scenario endpoints; frontend requires explicit mode switching UI ("Controlled Demonstration Mode" vs "Real Data Monitoring Mode"). |
| **Observability Telemetry UI** | `frontend/app/page.tsx` | **WORKING & NEEDS IMPROVEMENT** | Next.js dark glassmorphic dashboard with live polling, SVG reconstruction & KL charts, SHAP/KS tabs, and config modals. Needs UI panels for Registry & Gate. |

---

## 3. Current Architecture

The current system is organized into a modular Python backend and Next.js frontend:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              CURRENT RADAR ARCHITECTURE                                │
└────────────────────────────────────────────────────────────────────────────────────────┘

 [ Client / Simulator ] ──► POST /predict ──► [ ProductionClassifier ] ──► PredictResponse
                                 │
                         (Async Buffer List)
                                 │ (When len == 500)
                                 ▼
                     [ RadarOrchestrator Thread ]
                                 │
 ┌───────────────────────────────┴──────────────────────────────────────────────────────┐
 │ 7-Layer Monitoring Pipeline:                                                         │
 │  1. Ingestion & Preprocessing (MinMaxScaler)                                        │
 │  2. Raw Distribution Drift (VAE Reconstruction MSE + 1,000-Permutation KL + KS)       │
 │  3. Epistemic Uncertainty Estimation (3-Member LSTM RNN + MC Dropout)                │
 │  4. Multi-Signal Drift Fusion & Categorization (Gradual, Sudden, Covariate, Concept) │
 │  5. Model Health Score (MHS) Fusion (Healthy >0.85, Warning 0.65-0.85, Critical)     │
 │  6. Root Cause Attribution (SHAP KernelExplainer & Per-Feature KS Statistics)         │
 │  7. Adaptive Retraining (Borderline-SMOTE) -> Validation Gate -> SQLite Model Registry│
 └───────────────────────────────┬──────────────────────────────────────────────────────┘
                                 │ Write MonitoringResult
                                 ▼
                     [ In-Memory History Queue ]
                                 ▲
                                 │ Poll GET /monitoring/history (3s)
                     [ Next.js Telemetry UI ]
```

---

## 4. What Is Already Working

1. **Complete 7-Layer Detection Pipeline**: Ingestion $\rightarrow$ VAE Drift $\rightarrow$ RNN Uncertainty $\rightarrow$ Signal Fusion $\rightarrow$ MHS Calculation $\rightarrow$ SHAP/KS Root Cause $\rightarrow$ Retraining $\rightarrow$ Validation Gate $\rightarrow$ Registry.
2. **Asynchronous Ingestion**: FastAPI `/predict` returns instant predictions while queuing inputs for background monitoring.
3. **Statistical Permutation Testing**: 1,000-shuffle permutation test calculating mathematically exact $p$-values for KL divergence.
4. **Epistemic Uncertainty**: 3-LSTM RNN ensemble executing Monte Carlo Dropout to estimate model variance.
5. **Class Imbalance Oversampling**: Borderline-SMOTE pipeline active when minority class ratio drops below 20%.
6. **Pre-Deployment Safety Validation Gate**: Safety check comparing retrained model metrics (Accuracy, F1, Precision, Recall) against active production baselines before promotion.
7. **SQLite Model Registry**: Persistent DB logging model version tags (`v1`, `v2`), promotion statuses (`promoted`, `rejected`), rejection reasons, and evaluation metrics.
8. **Next.js Telemetry Dashboard**: Dark glassmorphic interface with custom SVG time-series charts, SHAP/KS root-cause tabs, and admin configuration controls.
9. **Master Verification Suite**: 16 dedicated stage test modules (`tests/test_stage1.py` through `test_stage16_frontend_api.py`) passing cleanly.

---

## 5. What Is Partially Working

1. **Operating Modes**: The backend supports scenario generation, but the UI lacks an explicit toggle to switch between **Controlled Demonstration Mode** and **Real Data Monitoring Mode**.
2. **Model Registry Binary Storage**: The SQLite DB tracks version metadata and metrics, but binary classifier model weights (`.joblib` / `.pkl`) are saved on disk without explicit database blob association.
3. **Supervised Streaming Detectors**: ADWIN and DDM are integrated into the fusion engine, but rely on synchronous ground-truth availability. A delayed label queueing system is needed.
4. **SHAP Attributions Rendering Speed**: SHAP `KernelExplainer` runs on background threads, but can take several seconds on larger feature windows. Performance caching or background sampling is needed.

---

## 6. Current Gaps

1. **Explicit UI Mode Switcher**: No dedicated UI bar to toggle between "Controlled Demonstration Mode" and "Real Data Monitoring Mode".
2. **Interactive Model Registry Lineage View**: The dashboard lacks a visual tab to inspect historical model versions, candidate validation results, and promotion/rejection logs.
3. **Validation Gate UI Modal**: When retraining occurs, the dashboard displays text alerts but lacks a dedicated side-by-side metric visualizer comparing Candidate vs Active baseline models.
4. **Ground-Truth Label Delay Queue**: System lacks an explicit delayed label arrival handler to model real-world scenarios where labels arrive hours or days after inference.
5. **Custom Dataset Replay Upload**: "Real Data Monitoring Mode" lacks a drag-and-drop CSV batch uploader allowing users to stream custom datasets through the radar.

---

## 7. Target Architecture

The target architecture enhances the system with dual operating modes, persistent telemetry logging, an explicit candidate validation gate UI, a dataset upload/replay pipeline, and binary artifact management in the model registry:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TARGET ENHANCED ARCHITECTURE                              │
└────────────────────────────────────────────────────────────────────────────────────────┘

                                ┌───────────────────────────────────────┐
                                │          OPERATING MODE SELECTOR      │
                                └───────────────────┬───────────────────┘
                                                    │
                      ┌─────────────────────────────┴─────────────────────────────┐
                      ▼                                                           ▼
      [ Controlled Demonstration Mode ]                           [ Real Data Monitoring Mode ]
   (Preset Scenarios: Normal, Gradual,                           (Custom CSV Upload, Batch Replay,
    Sudden, Imbalance, Recovery)                                  Stream Ingestion, Real Models)
                      │                                                           │
                      └─────────────────────────────┬─────────────────────────────┘
                                                    │
                                                    ▼
                                          [ API Server Gateway ]
                                         (POST /predict, POST /replay)
                                                    │
                                                    ▼
                                       [ Persistent Telemetry Store ]
                                          (SQLite / JSON Lines Log)
                                                    │
                                                    ▼
                                    [ 7-Layer Monitoring Orchestrator ]
                                                    │
        ┌───────────────────────────────────────────┼───────────────────────────────────────────┐
        ▼                                           ▼                                           ▼
[ Multi-Signal Fusion Engine ]            [ Root Cause Priority Grid ]               [ Adaptive Retraining ]
(VAE, KL, KS, Wasserstein, ADWIN, DDM)     (Side-by-Side SHAP + KS Matrix)          (Borderline-SMOTE + Fine-Tuning)
        │                                                                                       │
        ▼                                                                                       ▼
[ Model Health Score Engine ]                                                        [ Candidate Validation Gate ]
(Acc, Drift, Uncertainty, Stability)                                                 (F1 Margin & Acc Safety Check)
                                                                                                │
                                                                                 ┌──────────────┴──────────────┐
                                                                                 ▼                             ▼
                                                                           [ Promoted ]                  [ Rejected ]
                                                                                 │                             │
                                                                                 ▼                             ▼
                                                                     [ SQLite Registry vN+1 ]     [ Log Rejection & Retain vN ]
                                                                                 │                             │
                                                                                 └──────────────┬──────────────┘
                                                                                                │
                                                                                                ▼
                                                                                  [ Next.js Telemetry UI ]
                                                                                  • Executive Overview
                                                                                  • Live SVG Telemetry Charts
                                                                                  • SHAP vs. KS Priority Grid
                                                                                  • Validation Gate Modal
                                                                                  • Model Registry Lineage View
```

---

## 8. Recommended Technical Decisions

1. **Lightweight SQLite Registry Over MLflow**: Retain and expand the custom SQLite database (`models/model_registry.db`). It provides zero-dependency persistence, instant setup, and full schema control without requiring complex external MLflow servers.
2. **Side-by-Side SHAP + KS Attribution**: Maintain the dual explainability approach. KS tests answer *"Which features statistically changed?"*, while SHAP answers *"Which features drive model reconstruction error?"*. Combining them into a **Priority Matrix** prevents misleading interpretations.
3. **Dataset Replay Mechanism**: Implement batch replay for "Real Data Monitoring Mode" by reading CSV chunks sequentially at configurable time intervals, simulating real-world streaming inference cleanly.
4. **Borderline-SMOTE Oversampling**: Keep Borderline-SMOTE as the primary imbalance correction tool before retraining, as industrial IoT datasets (like SWaT) exhibit highly imbalanced anomaly distributions.
5. **Non-Blocking Background Threads**: Execute all background monitoring cycles, retraining, and SHAP calculations inside daemon threads to ensure inference response times remain under 15ms.

---

## 9. Two Operating Modes

### Mode 1: Controlled Demonstration Mode
*   **Purpose**: Evaluation, faculty presentations, and live benchmarks.
*   **Mechanics**: Allows controlled selection of preset scenarios:
    1. **Normal Production**: Stable data stream, green MHS (> 0.85), nominal metrics.
    2. **Gradual Drift**: Sensor readings slowly ramp over multiple batches; early warning alerts activate.
    3. **Sudden Drift**: Abrupt step-function shift triggering critical alerts ($p < 0.01$).
    4. **Class Imbalance**: Minority anomaly ratio drops to 3%; triggers Borderline-SMOTE logic.
    5. **Recovery**: Post-retraining nominal data showing MHS recovery back to healthy levels.
*   **UI Representation**: Prominently displays a yellow **"Controlled Demonstration Mode"** top badge with scenario controls.

### Mode 2: Real Data Monitoring Mode
*   **Purpose**: Real-world dataset monitoring and batch replay evaluation.
*   **Mechanics**: Allows users to:
    - Select or upload custom CSV datasets.
    - Select active classifier algorithms (Random Forest, XGBoost, MLP).
    - Replay batches sequentially with custom streaming speeds.
    - Monitor real-time distribution drift, uncertainty, and performance degradation.
*   **UI Representation**: Prominently displays a blue **"Real Data Monitoring Mode"** top badge with dataset stream controls.

---

## 10. Phased Implementation Roadmap

### Phase 0: Repository Audit and Baseline Verification
*   **Goal**: Establish complete verification baseline and ensure all 16 existing stage tests pass.
*   **Features**: System integrity audit, test verification.
*   **Files Likely to Change**: `tests/run_all_tests.py`, `README.md`.
*   **Dependencies**: Existing workspace dependencies (`scikit-learn`, `torch`, `fastapi`, `pandas`).
*   **Implementation Steps**:
    1. Run `python tests/run_all_tests.py` to confirm 100% test pass rate.
    2. Document existing baseline metrics and configuration parameters.
*   **Verification Steps**: Confirm master test suite exits with code 0.
*   **Expected Output**: Clean test execution output verifying all 16 stages.
*   **Failure Conditions**: Any failing stage test.
*   **Rollback Strategy**: Revert to clean baseline commit if files were modified.

### Phase 1: Telemetry Persistence & History Log Engine
*   **Goal**: Replace pure in-memory `monitoring_history` with persistent SQLite/JSON telemetry logging.
*   **Features**: Telemetry persistence across server restarts, historical query endpoints.
*   **Files Likely to Change**: `api/server.py`, `src/config.py`, `pipeline/orchestrator.py`.
*   **Dependencies**: SQLite standard library.
*   **Implementation Steps**:
    1. Create `TelemetryLogger` class writing `MonitoringResult` records to SQLite database `logs/telemetry.db`.
    2. Update `GET /monitoring/history` to read from persistent storage.
*   **Verification Steps**: Restart FastAPI server and verify historical monitoring batches persist in Next.js UI.
*   **Expected Output**: Telemetry records survive server restarts.
*   **Failure Conditions**: DB lock errors or serialization failures.
*   **Rollback Strategy**: Fallback to in-memory history buffer if DB write fails.

### Phase 2: Dual Operating Mode Selector & API Contracts
*   **Goal**: Formally introduce explicit system operating modes in API and config.
*   **Features**: `mode` parameter (`demo` vs `real_data`), mode transition endpoints.
*   **Files Likely to Change**: `src/config.py`, `api/server.py`, `src/simulator.py`.
*   **Dependencies**: FastAPI Pydantic models.
*   **Implementation Steps**:
    1. Add `operating_mode` to `RadarConfig` (`"demo"` | `"real_data"`).
    2. Expose `POST /admin/mode` endpoint to toggle operating mode.
    3. Update prediction endpoint metadata to tag responses with active operating mode.
*   **Verification Steps**: Test `POST /admin/mode` via `curl` / pytest and verify configuration state updates.
*   **Expected Output**: Clean mode switching without server restart.
*   **Failure Conditions**: Invalid mode string rejected with HTTP 400.
*   **Rollback Strategy**: Default to `"demo"` mode if invalid parameters passed.

### Phase 3: Enhanced Drift Signal Fusion & Priority Categorization
*   **Goal**: Refine drift type classification rules and evidence requirements.
*   **Features**: Categorize drift into `Covariate Drift`, `Concept Drift`, `Label Shift`, or `Uncertainty Warning`.
*   **Files Likely to Change**: `src/drift_fusion.py`, `pipeline/orchestrator.py`.
*   **Dependencies**: NumPy, SciPy.
*   **Implementation Steps**:
    1. Refine `DriftSignalFusionEngine` decision tree:
       - High VAE MSE + High KS ratio + Stable Labels $\rightarrow$ `Covariate Drift`.
       - High Supervised ADWIN/DDM + Accuracy Drop $\rightarrow$ `Concept Drift`.
       - High Epistemic Uncertainty + Low VAE MSE $\rightarrow$ `Novelty Warning`.
    2. Add `drift_classification_evidence` dictionary to `MonitoringResult`.
*   **Verification Steps**: Unit test each branch in `tests/test_stage7_fusion.py`.
*   **Expected Output**: Accurate drift categorization across all scenarios.
*   **Failure Conditions**: Misclassification of stable data as concept drift.
*   **Rollback Strategy**: Revert to basic weighted sum score if decision tree fails.

### Phase 4: Model Binary Persistence in SQLite Registry
*   **Goal**: Associate trained classifier binary artifacts directly with SQLite model registry records.
*   **Features**: Model binary artifact serialization (`.joblib`), versioned model artifact storage in `models/artifacts/`.
*   **Files Likely to Change**: `src/model_registry.py`, `src/classifier.py`.
*   **Dependencies**: `joblib`.
*   **Implementation Steps**:
    1. Add `artifact_path` column to `model_versions` table in `model_registry.db`.
    2. Update `log_version()` to serialize active model weights to `models/artifacts/{version_tag}.joblib`.
    3. Implement `load_model_version(version_tag)` method to restore historical models.
*   **Verification Steps**: Train a candidate model, verify `.joblib` saved on disk, and load historical version.
*   **Expected Output**: Full binary model reproducibility per registered version tag.
*   **Failure Conditions**: Corrupted binary file or serialization error.
*   **Rollback Strategy**: Retain current in-memory classifier weights if artifact loading fails.

### Phase 5: SHAP Optimization & Side-by-Side Priority Grid
*   **Goal**: Accelerate SHAP calculation speed and build a combined feature attribution priority grid.
*   **Features**: Background SHAP subsampling (100 samples max), combined Priority Score: $\text{Priority} = 0.5 \cdot \text{SHAP} + 0.5 \cdot (1 - p_{\text{KS}})$.
*   **Files Likely to Change**: `src/root_cause.py`, `pipeline/orchestrator.py`.
*   **Dependencies**: `shap`.
*   **Implementation Steps**:
    1. Subsample baseline reference set to 100 background samples for `KernelExplainer`.
    2. Implement `compute_priority_matrix(shap_dict, ks_results)` in `RootCauseEngine`.
*   **Verification Steps**: Run `test_stage9_root_cause.py` and benchmark execution time (< 2.0s per batch).
*   **Expected Output**: Fast, non-blocking root-cause attribution rendering.
*   **Failure Conditions**: SHAP computation timeout.
*   **Rollback Strategy**: Fallback to pure KS statistical ranking if SHAP calculation exceeds timeout.

### Phase 6: Controlled Demonstration Mode Enhancements
*   **Goal**: Expand scenario simulator to cover all 5 required demonstration workflows cleanly.
*   **Features**: `normal`, `gradual`, `sudden`, `imbalance`, and `recovery` scenario runners with clear status metadata.
*   **Files Likely to Change**: `src/simulator.py`, `api/server.py`.
*   **Dependencies**: Pandas, NumPy.
*   **Implementation Steps**:
    1. Verify parameter control for `drift_strength` (0.1 to 3.0).
    2. Expose `POST /admin/scenario` with payload `{"scenario": "gradual", "drift_strength": 1.5}`.
*   **Verification Steps**: Run `test_stage14_simulator.py` across all 5 scenarios.
*   **Expected Output**: Clear drift signal progression matching expected scenario behaviors.
*   **Failure Conditions**: Inconsistent data generation or NaN values.
*   **Rollback Strategy**: Reset to static CSV row reading.

### Phase 7: Real Data Monitoring Mode & Replay Engine
*   **Goal**: Enable custom CSV upload and step-by-step batch replay.
*   **Features**: `POST /admin/upload_dataset`, `POST /admin/replay_step` batch replay runner.
*   **Files Likely to Change**: `api/server.py`, `src/data_pipeline.py`.
*   **Dependencies**: `python-multipart`, Pandas.
*   **Implementation Steps**:
    1. Add file upload handler in `server.py` saving custom CSV to `data/uploads/`.
    2. Implement sliding window batch iterator streaming custom CSV rows through `/predict` buffer.
*   **Verification Steps**: Upload custom 1,000-row CSV file and execute replay stream.
*   **Expected Output**: Custom dataset monitored batch-by-batch cleanly.
*   **Failure Conditions**: Missing feature columns or incompatible data types.
*   **Rollback Strategy**: Validate schema on upload and reject incompatible CSV files.

### Phase 8: Candidate Model Validation Gate & Safety Workflow
*   **Goal**: Refine pre-deployment safety checks comparing candidate models vs active baselines.
*   **Features**: Validation Gate metrics breakdown (F1-score margin, minimum accuracy threshold, precision/recall stability).
*   **Files Likely to Change**: `src/validation_gate.py`, `pipeline/orchestrator.py`.
*   **Dependencies**: `scikit-learn`.
*   **Implementation Steps**:
    1. Enhance `ValidationGateEngine` to produce detailed metric comparison dictionaries.
    2. Log detailed validation gate evaluations to SQLite `model_versions` table (`promoted` vs `rejected`).
*   **Verification Steps**: Run `test_stage12_validation_gate.py` with degraded candidate model and verify rejection.
*   **Expected Output**: Candidate model rejected with explicit logged reason when performance drops.
*   **Failure Conditions**: Degraded candidate accidentally promoted.
*   **Rollback Strategy**: Hard rollback to active model baseline if gate check fails.

### Phase 9: Model Registry Visual Lineage & Metadata API
*   **Goal**: Expose model history endpoints for dashboard lineage visualization.
*   **Features**: `GET /registry/history`, `GET /registry/version/{tag}`, `POST /registry/rollback`.
*   **Files Likely to Change**: `api/server.py`, `src/model_registry.py`.
*   **Dependencies**: FastAPI endpoints.
*   **Implementation Steps**:
    1. Implement registry history endpoint returning list of all versions, metrics, and promotion statuses.
    2. Implement rollback endpoint allowing manual restoration of historical model versions.
*   **Verification Steps**: Test `GET /registry/history` via API test suite.
*   **Expected Output**: Complete version lineage JSON returned cleanly.
*   **Failure Conditions**: Version tag not found in database.
*   **Rollback Strategy**: Return active version metadata if requested tag is invalid.

### Phase 10: Telemetry UI Enhancements & Interactive Modals
*   **Goal**: Upgrade Next.js dashboard with Mode Switcher, Registry Lineage View, and Validation Gate Modal.
*   **Features**: Mode selector header, Validation Gate comparison modal, Model Registry lineage tab, Ground Truth status indicator.
*   **Files Likely to Change**: `frontend/app/page.tsx`, `frontend/app/hooks/useMonitoring.ts`.
*   **Dependencies**: React, Lucide Icons, TailwindCSS.
*   **Implementation Steps**:
    1. Add Mode Switcher component to top header ("Controlled Demonstration Mode" vs "Real Data Monitoring Mode").
    2. Build **Validation Gate Modal** rendering active vs candidate metrics side-by-side upon retraining.
    3. Build **Model Registry Lineage Tab** displaying version history table with promotion/rejection status badges.
    4. Add "Awaiting Ground Truth" badge when real labels are pending.
*   **Verification Steps**: Run `npm run build` in `frontend/` and inspect UI rendering.
*   **Expected Output**: Beautiful, responsive, dark glassmorphic dashboard with full observability panels.
*   **Failure Conditions**: Build errors or visual alignment issues.
*   **Rollback Strategy**: Revert layout modifications to previous clean `page.tsx` state.

### Phase 11: End-to-End System Testing & Benchmarking Suite
*   **Goal**: Conduct rigorous benchmark evaluation of drift detection latency, false positive rate, and retraining recovery.
*   **Features**: Automated benchmark script generating performance figures and summary metrics.
*   **Files Likely to Change**: `tests/benchmark_radar.py` [NEW].
*   **Dependencies**: Matplotlib / Pandas.
*   **Implementation Steps**:
    1. Create `tests/benchmark_radar.py` running 50 continuous batches across all scenarios.
    2. Calculate average detection latency (number of batches until $p < 0.01$), false positive rate on stable data, and MHS recovery delta.
*   **Verification Steps**: Run `python tests/benchmark_radar.py`.
*   **Expected Output**: Clean benchmark report confirming false positive rate $< 2\%$ and detection latency $< 2$ batches.
*   **Failure Conditions**: High false positive rate on nominal data.
*   **Rollback Strategy**: Adjust fusion vote thresholds in `src/config.py`.

### Phase 12: Final Presentation & Demonstration Workflow
*   **Goal**: Package the system for a seamless 5–10 minute live presentation.
*   **Features**: Pre-flight demo script (`run_demo_scenario.py`), reset controls, slide-ready architecture diagrams.
*   **Files Likely to Change**: `demo_script.py` [NEW], `README.md`.
*   **Dependencies**: Python standard library.
*   **Implementation Steps**:
    1. Create automated demo script driving the system through the 12-step presentation story.
    2. Add single-command launcher for demo presentation (`./run.sh --demo`).
*   **Verification Steps**: Execute full 12-step demo flow without manual intervention.
*   **Expected Output**: Flawless live demonstration demonstrating complete model decay lifecycle.
*   **Failure Conditions**: Demo script crash or API timeout.
*   **Rollback Strategy**: Provide manual step-by-step backup instructions in `README.md`.

---

## 11. Testing and Verification Strategy

Every phase must undergo explicit multi-level verification before being marked complete:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              VERIFICATION FRAMEWORK                                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│  1. Unit Tests          → PyTest / Unittest for core math & algorithms                 │
│  2. Integration Tests   → FastAPI TestClient verifying REST endpoints                  │
│  3. Stage Verification  → Master runner tests/run_all_tests.py (16 stages)             │
│  4. UI Build Verification → Next.js production build check (npm run build)             │
│  5. Scenario Benchmarks → Benchmark script verifying detection latency & false positives│
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Mandatory Verification Matrix
- **Nominal Data Test**: Ingest `current_stable` window. Expected: $p$-value $> 0.05$, MHS $> 0.85$, Status `Healthy`.
- **Gradual Drift Test**: Ingest `current_drift` window with ramp. Expected: $p$-value drops, MHS enters `Warning` ($0.65 - 0.85$), top drifted feature identified as `LIT101`.
- **Sudden Drift Test**: Ingest step-drift batch. Expected: $p < 0.01$, MHS drops to `Critical` ($< 0.65$), retraining triggered.
- **Validation Gate Rejection Test**: Inject degraded candidate model. Expected: Validation Gate rejects candidate, logs rejection reason, retains active baseline model.
- **Candidate Promotion Test**: Retrain with SMOTE on drifted batch. Expected: Validation Gate passes candidate, promotes to next version (`v2`), MHS recovers to `Healthy`.

---

## 12. Dataset and Demonstration Strategy

### Primary Dataset: SWaT (Secure Water Treatment) Synthetic IoT Dataset
*   **Source**: Industrial IoT water treatment facility testbed (`data/synthetic_swat.csv`).
*   **Dimensions**: 51 continuous physical sensor metrics (Flow rate `FIT`, Level `LIT`, Pressure `PIT`, Differential Pressure `DPIT`, Actuator states) + 1 binary Label (`0` = Nominal, `1` = Anomaly).
*   **Data Partitioning**:
    1. `reference` (20,000 samples): Baseline training set for scalers, VAE, RNN, and initial classifier.
    2. `current_stable` (10,000 samples): Nominal production stream.
    3. `current_drift` (8,000 samples): Sensor drift injected into `LIT101`, `DPIT301`, and `P402`.

### Secondary Dataset (For Real Data Monitoring Mode): Synthetic Credit Fraud Dataset
*   **Source**: Tabular transaction monitoring dataset (`data/synthetic_fraud.csv`).
*   **Purpose**: Demonstrates system adaptability to financial tabular domain without modifying monitoring pipeline logic.

---

## 13. Dashboard Roadmap

The Next.js dashboard (`frontend/app/page.tsx`) will be refined across the following structured layout:

1. **Header Navigation Bar**:
   - Project Title & Subtitle.
   - Operating Mode Selector Pill ("Controlled Demonstration Mode" vs "Real Data Monitoring Mode").
   - System Health Indicator Badge (`Connected` / `Disconnected`).
   - Active Model Version Tag (`v1`, `v2`, etc.).
2. **Executive KPI Cards Grid**:
   - Card 1: **Model Health Score (MHS)** Gauge & Status (`Healthy`, `Warning`, `Critical`).
   - Card 2: **Drift Severity Index** & KL $p$-value ($p < 0.01$ threshold).
   - Card 3: **Epistemic Uncertainty** Score from Monte Carlo RNN Ensemble.
   - Card 4: **Model Performance** (Accuracy/F1) & Ground-Truth Label Status Badge ("Labels Available" / "Awaiting Ground Truth").
3. **Telemetry Visualization Charts**:
   - **Chart A**: Reconstruction Error (MSE) vs Dynamic Threshold over time.
   - **Chart B**: Statistical KL Divergence Trajectory & Significance Cutoff ($p = 0.01$).
4. **Explainability & Root Cause Attribution Panel**:
   - Dual Tab Switcher: **SHAP Feature Attributions** vs **Per-Feature KS Statistics**.
   - Priority Matrix view highlighting root-cause sensors (`LIT101`, `DPIT301`, `P402`).
5. **Model Lifecycle & Validation Gate Panel**:
   - Retraining Status & Candidate Model Validation Gate Modal.
   - Model Registry Lineage Table (Version history, promotion/rejection reasons, F1 scores).
6. **Scenario Simulator Controls** (Visible only in Controlled Demonstration Mode):
   - Scenario Selector Dropdown (`Normal`, `Gradual Drift`, `Sudden Drift`, `Class Imbalance`, `Recovery`).
   - Drift Strength Slider ($0.5\times$ to $3.0\times$).
   - Action Buttons: `[ Simulate Batch ]`, `[ Trigger Retrain ]`, `[ Reset Radar ]`.

---

## 14. Model Lifecycle and Retraining Strategy

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          MODEL ADAPTATION & RETRAINING FLOW                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────┐
 │   Active Model (v1)     │
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Monitoring Cycle Runs   │ ──► MHS Drops to Critical (< 0.65)
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Retraining Triggered    │ ──► Extract 80% Historical Reference + 20% Drifted Batch
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Class Imbalance Check   │ ──► Apply Borderline-SMOTE if Anomaly Ratio < 20%
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Fit Candidate Model     │ ──► Train fresh Candidate Classifier instance
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │ Candidate Validation    │
 │ Gate Evaluation         │ ──► Evaluate Active vs Candidate on Validation Split
 └────────────┬────────────┘
              │
       ┌──────┴───────────────────────────┐
       ▼                                  ▼
[ Candidate F1 >= Active F1 - 0.02 ]   [ Candidate F1 < Active F1 - 0.02 ]
       │                                  │
       ▼                                  ▼
┌──────────────────────────┐     ┌──────────────────────────┐
│ Promote Candidate to v2  │     │ Reject Candidate Model   │
│ Log Version in SQLite DB │     │ Log Rejection in DB      │
│ Set Active Model = v2    │     │ Retain Active Model (v1) │
└──────────────────────────┘     └──────────────────────────┘
```

---

## 15. Benchmarking and Evaluation

The platform will be benchmarked on quantitative operational performance:

| Benchmark Metric | Target Threshold | Evaluation Method |
| :--- | :--- | :--- |
| **Drift Detection Latency** | $< 2$ batches ($1,000$ samples) | Batches elapsed from drift injection to $p < 0.01$ confirmation. |
| **False Positive Rate** | $< 2.0\%$ | Rate of drift alerts triggered during `current_stable` nominal stream. |
| **Root-Cause Attribution Precision** | $\ge 90.0\%$ | Percentage of top 3 SHAP/KS features matching ground-truth drifted sensors (`LIT101`, `DPIT301`, `P402`). |
| **Retraining Recovery Speed** | $< 5.0$ seconds | Execution time to execute Borderline-SMOTE, fine-tune VAE/Classifier, and evaluate Validation Gate. |
| **MHS Recovery Delta** | $\ge +0.25$ MHS gain | Increase in Model Health Score after candidate model promotion on drifted stream. |
| **API Inference Latency** | $< 15$ ms / request | Response latency for `POST /predict` under continuous load. |

---

## 16. Risks and Simplifications

1. **Risk: Computationally Expensive SHAP Execution**
   - *Mitigation*: Subsample background reference dataset to 100 points for `KernelExplainer` and execute SHAP on background worker threads so API predictions remain sub-15ms.
2. **Risk: False Alarms on Small Batch Sizes**
   - *Mitigation*: Set default window size to 500 samples and enforce dual-confirmation (reconstruction error threshold + 1,000-permutation KL $p < 0.01$).
3. **Risk: Overfitting Candidate Model During Fine-Tuning**
   - *Mitigation*: Enforce strict 80/20 data ratio (80% historical reference baseline + 20% recent drifted data) during retraining, combined with pre-deployment Validation Gate enforcement.
4. **Simplification: Local In-Memory Buffer & SQLite Persistence**
   - *Rationale*: Avoids complex Apache Kafka or Kubernetes deployment overhead for a major undergraduate project while maintaining identical MLOps architectural principles.

---

## 17. Final Presentation Workflow (5–10 Minute Live Story)

The demonstration will follow a narrative demonstrating the complete lifecycle of model decay and recovery:

- **STEP 1 (0:00 - 0:45)**: *Introduction & System Architecture*. Present deployed industrial model monitoring continuous water treatment IoT sensors.
- **STEP 2 (0:45 - 1:30)**: *Healthy Baseline Monitoring*. Stream `normal` batch. Show Next.js dashboard displaying green `Healthy` status (MHS = 0.94), low reconstruction error, and $p > 0.05$.
- **STEP 3 (1:30 - 2:30)**: *Controlled Drift Injection*. Trigger `gradual` sensor drift scenario. Observe reconstruction error rising on live SVG chart and Model Health Score dropping to `Warning` (0.74).
- **STEP 4 (2:30 - 3:30)**: *Multi-Signal Drift Confirmation*. Trigger `sudden` step drift. Show KL Permutation test yielding $p = 0.0001 < 0.01$. Alert badge changes to critical red `Critical`.
- **STEP 5 (3:30 - 4:30)**: *Epistemic Uncertainty Detection*. Highlight Monte Carlo RNN Ensemble output showing elevated uncertainty (0.78), confirming out-of-distribution novelty.
- **STEP 6 (4:30 - 5:30)**: *Explainable Root-Cause Attribution*. Switch to SHAP & KS attribution tabs. Highlight top 3 drifted features (`LIT101`, `DPIT301`, `P402`) explaining *why* the model decayed.
- **STEP 7 (5:30 - 6:30)**: *Automated Adaptive Retraining*. Show system triggering automated retraining with Borderline-SMOTE oversampling as MHS drops below 0.65.
- **STEP 8 (6:30 - 7:30)**: *Pre-Deployment Validation Gate*. Open Candidate Validation Gate Modal. Compare candidate model metrics against active baseline model.
- **STEP 9 (7:30 - 8:30)**: *Model Registry Promotion*. Show Candidate model passing safety checks and being promoted to version `v2` in SQLite Model Registry.
- **STEP 10 (8:30 - 9:15)**: *System Recovery*. Stream post-retraining recovery batch. Show Model Health Score recovering to green `Healthy` (0.91).
- **STEP 11 (9:15 - 9:45)**: *Real Data Monitoring Mode Demonstration*. Toggle to Real Data Monitoring Mode, upload custom dataset CSV, and demonstrate batch replay.
- **STEP 12 (9:45 - 10:00)**: *Conclusion & Q&A*. Summarize MLOps impact: silent failure detection, explainable attribution, safety-gated model adaptation.

---

## 18. Four-Person Team Collaboration Plan

To maximize productivity and eliminate git merge conflicts, work is divided across 4 distinct ownership domains with clear file boundaries:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           FOUR-PERSON TEAM RESPONSIBILITIES                            │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MEMBER 1: ML & Drift Engine Architect          → src/autoencoder.py, calibration.py,   │
│                                                   drift_fusion.py, uncertainty.py      │
│ MEMBER 2: Model Lifecycle & Retraining Lead    → src/classifier.py, model_registry.py,  │
│                                                   validation_gate.py, imbalance_handler.py│
│ MEMBER 3: Backend Systems & Orchestration Lead → api/server.py, pipeline/orchestrator.py│
│                                                   src/simulator.py, src/config.py      │
│ MEMBER 4: Frontend & Telemetry UI Developer    → frontend/app/page.tsx, hooks/,       │
│                                                   components/, globals.css             │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Module Ownership & File Boundaries

#### Member 1: ML & Drift Engine Architect
*   **Ownership**: Core drift algorithms, VAE, statistical tests, uncertainty, and signal fusion.
*   **Primary Files**:
    - [autoencoder.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/autoencoder.py)
    - [calibration.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/calibration.py)
    - [drift_fusion.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/drift_fusion.py)
    - [uncertainty.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/uncertainty.py)
*   **Tests Owned**: `test_stage3_vae_kl.py`, `test_stage4_wasserstein_ks.py`, `test_stage5_adwin_ddm.py`, `test_stage6_uncertainty.py`, `test_stage7_fusion.py`.

#### Member 2: Model Lifecycle & Retraining Lead
*   **Ownership**: Multi-classifier backend, SMOTE imbalance oversampling, Validation Gate, and SQLite Model Registry.
*   **Primary Files**:
    - [classifier.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/classifier.py)
    - [imbalance_handler.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/imbalance_handler.py)
    - [validation_gate.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/validation_gate.py)
    - [model_registry.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/model_registry.py)
*   **Tests Owned**: `test_stage10_multiclassifier.py`, `test_stage11_retraining_smote.py`, `test_stage12_validation_gate.py`, `test_stage13_versioning_sqlite.py`.

#### Member 3: Backend Systems & Orchestration Lead
*   **Ownership**: FastAPI REST API server, pipeline orchestrator, configuration manager, scenario simulator, telemetry persistence.
*   **Primary Files**:
    - [server.py](file:///d:/model_decay_radar_v2/model_decay_radar/api/server.py)
    - [orchestrator.py](file:///d:/model_decay_radar_v2/model_decay_radar/pipeline/orchestrator.py)
    - [simulator.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/simulator.py)
    - [config.py](file:///d:/model_decay_radar_v2/model_decay_radar/src/config.py)
*   **Tests Owned**: `test_stage1.py`, `test_stage2_predict_api.py`, `test_stage14_simulator.py`, `test_stage15_modes.py`.

#### Member 4: Frontend & Telemetry UI Developer
*   **Ownership**: Next.js dashboard, SVG telemetry charts, SHAP/KS root-cause tabs, Validation Gate modal, Registry lineage view, Operating Mode switcher.
*   **Primary Files**:
    - `frontend/app/page.tsx`
    - `frontend/app/hooks/useMonitoring.ts`
    - `frontend/app/layout.tsx`
    - `frontend/app/globals.css`
*   **Tests Owned**: `test_stage6_frontend.py`, `test_stage16_frontend_api.py`.

### Collaboration Rules & Git Protocol
1. **Strict Core Module Ownership**: No team member may modify files outside their primary domain without explicit code review approval.
2. **Frozen Data Schemas**: The `MonitoringResult` dataclass in `pipeline/orchestrator.py` serves as the single source of truth contract between backend and frontend. Schema modifications require team sign-off.
3. **Feature Branch Strategy**:
   - `feature/ml-drift-engine` (Member 1)
   - `feature/model-lifecycle` (Member 2)
   - `feature/backend-api` (Member 3)
   - `feature/frontend-ui` (Member 4)
4. **Integration Checkpoints**: Merge all feature branches into `main` only at designated phase completion checkpoints after running `python tests/run_all_tests.py`.

---

## 19. Definition of Done

A phase or feature is considered **DONE** only when:

1. **Code Implementation**: All specified Python and TypeScript components are fully implemented without stubbed returns or mock fallbacks.
2. **Automated Test Verification**: The master test runner (`python tests/run_all_tests.py`) passes 100% cleanly without errors or warnings.
3. **End-to-End Execution**: Incoming prediction payloads process through the 7-layer pipeline and render updated telemetry on the Next.js UI in real time.
4. **UI Build Success**: The Next.js production build (`npm run build`) executes cleanly with zero TypeScript errors.
5. **Documentation Alignment**: Any schema or API additions are reflected in `MODEL_DECAY_DETECTION.md` and `README.md`.
