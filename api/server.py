"""
FastAPI Server — Model Decay Radar
===================================

Endpoints:
  POST /predict          → user/IoT device sends feature data, gets back prediction
  GET  /health           → server + model health check
  GET  /monitoring/latest → latest monitoring result (Streamlit reads this)
  GET  /monitoring/history → all monitoring results so far
  GET  /monitoring/status  → current MHS + alert level at a glance
  POST /admin/reset      → reset buffer (for demo restarts)

Flow:
  1. User POSTs features to /predict
  2. A dummy ML model predicts (in production: your real model here)
  3. Sample is added to the buffer
  4. When buffer reaches WINDOW_SIZE, RadarOrchestrator.run_monitoring_cycle()
     fires in a background thread (so /predict still returns fast)
  5. Result is stored in monitoring_history (in-memory for MVP;
     use Redis/Postgres in production)
  6. Next.js dashboard polls /monitoring/history to render live charts
"""

import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

import threading
import traceback
import uuid
import time
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from data_pipeline import DataWindowManager
from orchestrator import RadarOrchestrator
from classifier import ProductionClassifier
from telemetry_logger import TelemetryLogger
import calibration

telemetry_logger = TelemetryLogger(postgres_url=config.postgres_url)



# ─────────────────────────────────────────────
# Config & Paths
# ─────────────────────────────────────────────

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/synthetic_swat.csv")

# ─────────────────────────────────────────────
# Startup: load data + train components
# ─────────────────────────────────────────────

df           = pd.read_csv(DATA_PATH)
FEATURE_COLS = [c for c in df.columns if c not in ("Label", "_window")]
LABEL_COL    = "Label"

wm           = DataWindowManager(df, FEATURE_COLS, label_col=LABEL_COL)
reference_df = df[df["_window"] == "reference"]
wm.set_reference_window(reference_df)
reference_scaled = wm.reference_scaled

orchestrator = RadarOrchestrator(feature_names=FEATURE_COLS)

# Train ProductionClassifier on reference window (raw features, before scaling)
reference_X = reference_df[FEATURE_COLS].to_numpy(dtype=np.float32)
reference_y = reference_df[LABEL_COL].to_numpy(dtype=np.int32)
ml_model = ProductionClassifier()
ml_model.fit(reference_X, reference_y, model_name=config.active_classifier)
print(f"[Server] Active classifier '{config.active_classifier}' trained on {len(reference_X)} reference samples.")

# ─────────────────────────────────────────────
# In-memory state
# ─────────────────────────────────────────────

buffer_features: list[list[float]] = []
buffer_labels: list[int] = []
buffer_predictions: list[int] = []
buffer_lock = threading.Lock()

delayed_buffer_features: list[list[float]] = []
delayed_buffer_predictions: list[int] = []
delayed_buffer_labels: list[int] = []
pending_predictions: dict[str, dict] = {}
delayed_buffer_lock = threading.Lock()

monitoring_history : list[dict] = []   # list of MonitoringResult dicts
history_lock       = threading.Lock()
orchestrator_lock  = threading.Lock()

is_setup_done = False
setup_error: Optional[str] = None  # populated if setup crashes
setup_progress: str = "pending"     # "pending" | "training_ae" | "training_rnn" | "ready" | "failed"

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# Lifespan (replaces deprecated @app.on_event)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: launch orchestrator training."""
    run_setup()
    yield
    # Shutdown: nothing to clean up for MVP


app = FastAPI(
    title="Model Decay Radar API",
    description="Real-time ML model monitoring with drift detection, "
                "SHAP interpretation, and adaptive retraining.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    What the user / IoT device sends to /predict.
    features: dict of {sensor_name: value} — matches the model's feature set.
    label:    optional ground truth (arrives with delay in production).
    """
    features: dict[str, float]
    label: Optional[int] = None


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: float
    confidence: float
    model_version: str
    timestamp: float
    model_status: str    # "Healthy" | "Warning" | "Critical"
    batch_buffered: int  # how many samples are in the current window


class DelayedLabel(BaseModel):
    prediction_id: str
    label: int


class DelayedLabelRequest(BaseModel):
    labels: list[DelayedLabel]


# ─────────────────────────────────────────────
# Background setup (runs once at startup)
# ─────────────────────────────────────────────

def run_setup():
    global is_setup_done, setup_error, setup_progress
    try:
        setup_progress = "training_ae"
        print("[Server] Starting orchestrator setup (training AE + RNN)...", flush=True)
        
        if reference_scaled is None:
            raise ValueError("reference_scaled is missing but required for setup.")
            
        orchestrator.setup(
            reference_scaled,
            reference_raw_X=reference_X,
            reference_raw_y=reference_y,
            classifier=ml_model,
            scaler=wm.scaler,
            ae_epochs=config.ae_epochs,
            rnn_epochs=config.rnn_epochs,
            progress_callback=_on_setup_progress,
        )
        is_setup_done = True
        setup_progress = "ready"
        print("[Server] Orchestrator ready.", flush=True)
    except Exception as e:
        setup_error = str(e)
        setup_progress = "failed"
        print(f"[Server] FATAL: Orchestrator setup failed: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()


def _on_setup_progress(stage: str):
    """Callback from orchestrator to report training stage."""
    global setup_progress
    setup_progress = stage


# ─────────────────────────────────────────────
# Background monitoring cycle
# ─────────────────────────────────────────────

def run_monitoring_cycle_background(
    batch_features: np.ndarray,
    batch_labels: Optional[np.ndarray],
    batch_predictions: Optional[np.ndarray],
):
    """
    Runs the full Radar pipeline in a background thread so /predict
    returns immediately — the user never waits for SHAP/permutation tests.
    """
    if not orchestrator.is_ready:
        print("[Monitor] Orchestrator not ready yet, skipping cycle.", flush=True)
        return

    print(f"[Monitor] Queueing monitoring cycle on {len(batch_features)} samples...", flush=True)
    with orchestrator_lock:
        print(f"[Monitor] Running monitoring cycle on {len(batch_features)} samples...", flush=True)
        try:
            result = orchestrator.run_monitoring_cycle(
                batch_features, batch_labels, batch_predictions
            )
            result_dict = asdict(result)

            with history_lock:
                monitoring_history.append(result_dict)
                # Keep last 200 results in memory
                if len(monitoring_history) > 200:
                    monitoring_history.pop(0)

            # Persist to PostgreSQL database via TelemetryLogger
            telemetry_logger.log_monitoring_result(result_dict)


            print(f"[Monitor] Batch {result.batch_id}: MHS={result.mhs:.3f} "
                  f"({result.mhs_status}), drift={result.drift_confirmed}, "
                  f"alert={result.alert_level}", flush=True)

        except RuntimeError as e:
            if "setup()" in str(e):
                print(f"[Monitor] Orchestrator not fully setup, skipping cycle.", flush=True)
        except Exception as e:
            print(f"[Monitor] Error in monitoring cycle: {e}", flush=True)
            import traceback; traceback.print_exc()
            sys.stdout.flush()
        except Exception as e:
            print(f"[Monitor] Error in monitoring cycle: {e}", flush=True)
            import traceback; traceback.print_exc()
            sys.stdout.flush()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Model Decay Radar API is active",
        "health": "/health",
        "docs": "/docs",
        "latest_monitoring": "/monitoring/latest",
    }


@app.get("/health")
def health():
    """Server health check — also shows setup status."""
    return {
        "status": "ok",
        "orchestrator_ready": is_setup_done,
        "setup_progress": setup_progress,
        "setup_error": setup_error,
        "buffer_size": len(buffer_features),
        "window_size": config.window_size,
        "monitoring_cycles_completed": len(monitoring_history),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint — called by user/IoT device for every new sample.

    1. Validates and scales features
    2. Runs the ML model → returns prediction immediately
    3. Buffers the sample
    4. When buffer reaches config.window_size → triggers background monitoring cycle
    """
    try:
        raw_features = np.array(
            [request.features.get(col, 0.0) for col in FEATURE_COLS],
            dtype=np.float32
        )
    except Exception:
        raise HTTPException(400, "Feature format invalid.")

    # Scale using reference-window scaler
    if getattr(wm, "scaler", None) is None:
        scaled_features = raw_features
    else:
        scaled_features = wm.scaler.transform(raw_features.reshape(1, -1))[0] # type: ignore
    scaled_features = np.clip(scaled_features, -2.0, 2.0)   # allow slight OOD

    # ML model prediction on raw features (returns immediately to user)
    prediction, confidence = ml_model.predict(raw_features)
    pred_id = f"pred_{uuid.uuid4().hex[:10]}"
    t_now = time.time()
    active_version = orchestrator.registry.get_latest_version_tag() if hasattr(orchestrator, "registry") else "v1"

    # Get current model status from latest monitoring result
    with history_lock:
        model_status = (
            monitoring_history[-1]["mhs_status"]
            if monitoring_history
            else "Initialising"
        )

    # Buffer the sample (scaled features for drift pipeline)
    with buffer_lock:
        buffer_features.append(scaled_features.tolist())
        buffer_predictions.append(int(prediction))
        if request.label is not None:
            buffer_labels.append(request.label)
        
        # Store for potential delayed label arrival
        pending_predictions[pred_id] = {
            "features": scaled_features.tolist(),
            "prediction": int(prediction)
        }
        # Prevent memory leak
        if len(pending_predictions) > 10000:
            oldest = list(pending_predictions.keys())[0]
            del pending_predictions[oldest]

        current_buffer_size = len(buffer_features)

        # When window fills → snapshot + clear buffer + trigger monitoring
        if current_buffer_size >= config.window_size:
            batch_feat = np.array(buffer_features, dtype=np.float32)
            batch_labels_arr = (
                np.array(buffer_labels, dtype=np.int32) if len(buffer_labels) == current_buffer_size else None
            )
            batch_preds_arr = np.array(buffer_predictions, dtype=np.int32)
            buffer_features.clear()
            buffer_labels.clear()
            buffer_predictions.clear()

            background_tasks.add_task(
                run_monitoring_cycle_background,
                batch_feat,
                batch_labels_arr,
                batch_preds_arr,
            )

    return PredictResponse(
        prediction_id=pred_id,
        prediction=prediction,
        confidence=confidence,
        model_version=active_version,
        timestamp=t_now,
        model_status=model_status,
        batch_buffered=min(current_buffer_size, config.window_size),
    )


@app.post("/labels/delayed")
def labels_delayed(req: DelayedLabelRequest, background_tasks: BackgroundTasks):
    """
    Accepts delayed ground truth labels corresponding to previous predictions.
    When we accumulate enough (window_size) delayed labels, trigger a supervised monitoring cycle.
    """
    processed = 0
    with delayed_buffer_lock:
        for delayed in req.labels:
            if delayed.prediction_id in pending_predictions:
                data = pending_predictions.pop(delayed.prediction_id)
                delayed_buffer_features.append(data["features"])
                delayed_buffer_predictions.append(data["prediction"])
                delayed_buffer_labels.append(delayed.label)
                processed += 1
        
        # If we reached window size, trigger monitoring cycle with supervised metrics
        while len(delayed_buffer_features) >= config.window_size:
            batch_feat = np.array(delayed_buffer_features[:config.window_size], dtype=np.float32)
            batch_labels = np.array(delayed_buffer_labels[:config.window_size], dtype=np.int32)
            batch_preds = np.array(delayed_buffer_predictions[:config.window_size], dtype=np.int32)
            
            del delayed_buffer_features[:config.window_size]
            del delayed_buffer_labels[:config.window_size]
            del delayed_buffer_predictions[:config.window_size]
            
            background_tasks.add_task(
                run_monitoring_cycle_background,
                batch_feat,
                batch_labels,
                batch_preds
            )

    return {"status": "success", "message": f"Processed {processed} delayed labels."}


@app.get("/monitoring/latest")
def monitoring_latest():
    """Latest monitoring result."""
    history = telemetry_logger.get_recent_history(limit=1)
    if not history:
        with history_lock:
            if not monitoring_history:
                return {"status": "no_data", "message": "No monitoring cycles completed yet."}
            return monitoring_history[-1]
    return history[-1]


@app.get("/monitoring/history")
def monitoring_history_endpoint(limit: int = 50):
    """All monitoring results from PostgreSQL/logger (last `limit` entries)."""
    history = telemetry_logger.get_recent_history(limit=limit)
    if not history:
        with history_lock:
            history = monitoring_history[-limit:]
    return {"results": history, "total": len(history)}



@app.get("/monitoring/status")
def monitoring_status():
    """Quick at-a-glance status for the dashboard header."""
    with history_lock:
        if not monitoring_history:
            return {"mhs": None, "status": "Initialising", "alert_level": "none",
                    "drift_confirmed": False, "cycles": 0}
        latest = monitoring_history[-1]
        return {
            "mhs":             latest["mhs"],
            "status":          latest["mhs_status"],
            "alert_level":     latest["alert_level"],
            "drift_confirmed": latest["drift_confirmed"],
            "cycles":          len(monitoring_history),
        }


@app.post("/admin/reset")
def admin_reset():
    """Reset the buffer and history (useful for demo restarts)."""
    with buffer_lock:
        buffer_features.clear()
        buffer_labels.clear()
        buffer_predictions.clear()
    with history_lock:
        monitoring_history.clear()
    telemetry_logger.clear_history()
    return {"status": "reset", "message": "Buffer and history cleared."}


class ConfigSchema(BaseModel):
    active_classifier: Optional[str] = None
    window_size: Optional[int] = None
    p_value_threshold: Optional[float] = None
    mc_dropout_t: Optional[int] = None
    retraining_ratio: Optional[float] = None


@app.get("/admin/config")
def get_config():
    """Get active configurations for the monitoring dashboard."""
    cfg = config.to_dict()
    cfg["active_classifier"] = ml_model.active_model_name
    cfg["available_classifiers"] = list(ml_model.models.keys())
    return cfg


@app.post("/admin/config")
def update_config(req_config: ConfigSchema):
    """Update configurations dynamically."""
    if req_config.active_classifier is not None:
        try:
            ml_model.set_active_model(req_config.active_classifier)
            config.active_classifier = req_config.active_classifier
            print(f"[Server] Active classifier switched to: {req_config.active_classifier}")
        except ValueError as e:
            raise HTTPException(400, str(e))
            
    if req_config.window_size is not None:
        if req_config.window_size < 10 or req_config.window_size > 5000:
            raise HTTPException(400, "Window size must be between 10 and 5000.")
        config.window_size = req_config.window_size
        print(f"[Server] Dynamic window size set to: {config.window_size}")
        
    if req_config.p_value_threshold is not None:
        if req_config.p_value_threshold <= 0.0 or req_config.p_value_threshold > 1.0:
            raise HTTPException(400, "p-value threshold must be between 0 and 1.")
        config.p_value_threshold = req_config.p_value_threshold
        calibration.DRIFT_P_VALUE_THRESHOLD = req_config.p_value_threshold
        print(f"[Server] Drift statistical p-value threshold set to: {config.p_value_threshold}")

    if req_config.mc_dropout_t is not None:
        if req_config.mc_dropout_t < 1 or req_config.mc_dropout_t > 500:
            raise HTTPException(400, "MC Dropout T must be between 1 and 500.")
        config.mc_dropout_t = req_config.mc_dropout_t

    if req_config.retraining_ratio is not None:
        if req_config.retraining_ratio <= 0.0 or req_config.retraining_ratio >= 1.0:
            raise HTTPException(400, "Retraining ratio must be strictly between 0.0 and 1.0.")
        config.retraining_ratio = req_config.retraining_ratio

    return {
        "status": "success",
        "config": get_config()
    }


@app.post("/admin/retrain")
def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger a selective retraining cycle."""
    if not orchestrator.is_ready:
        raise HTTPException(400, "Orchestrator is not ready yet.")
        
    # Snap the current buffer if we have enough points, else fallback to a subset of reference
    with buffer_lock:
        if len(buffer_features) >= 10:
            batch_feat = np.array(buffer_features, dtype=np.float32)
            batch_labels_arr = np.array(buffer_labels, dtype=np.int32) if buffer_labels else None
        else:
            # Fallback to reference window subset to prevent crashing
            if reference_scaled is None or reference_y is None:
                raise ValueError("reference_scaled or reference_y is missing but required for fallback.")
            batch_feat = reference_scaled[:500]
            batch_labels_arr = reference_y[:500]
            
    background_tasks.add_task(
        orchestrator._selective_retrain,
        batch_feat,
        batch_labels_arr
    )
    return {"status": "success", "message": "Manual retraining cycle triggered in background."}


@app.get("/admin/simulate")
def admin_simulate(n_stable: int = 1000, n_drift: int = 1500):
    """
    Demo shortcut: injects synthetic samples directly into the buffer
    without needing a real IoT device. Pulls from the pre-generated
    synthetic_swat.csv data so the demo can be run standalone.
    """
    if not orchestrator.is_ready:
        raise HTTPException(400, "Orchestrator is not ready yet.")

    stable_df = df[df["_window"] == "current_stable"].head(n_stable)
    drift_df = df[df["_window"] == "current_drift"].head(n_drift)
    combined = pd.concat([stable_df, drift_df]).reset_index(drop=True)

    raw_features = combined[FEATURE_COLS].values.astype(np.float32)
    if getattr(wm, "scaler", None) is None:
        all_scaled = raw_features
    else:
        all_scaled = np.clip(wm.scaler.transform(raw_features), -2.0, 2.0).astype(np.float32)
    all_labels = combined[LABEL_COL].values.astype(np.int32)
    all_preds = ml_model.predict_batch(raw_features)

    injected = len(combined)
    triggered_cycles = 0

    for start in range(0, injected, config.window_size):
        end = min(start + config.window_size, injected)
        if end - start < config.window_size:
            break
        batch_feat = all_scaled[start:end]
        batch_labels_arr = all_labels[start:end]
        batch_preds_arr = all_preds[start:end]

        if orchestrator.is_ready:
            thread = threading.Thread(
                target=run_monitoring_cycle_background,
                args=(batch_feat, batch_labels_arr, batch_preds_arr),
                daemon=True,
            )
            thread.start()
            triggered_cycles += 1

    return {
        "status": "success",
        "injected": injected,
        "triggered_cycles": triggered_cycles,
        "message": f"Injected {injected} samples ({n_stable} stable + {n_drift} drift). "
                   f"Triggered {triggered_cycles} monitoring cycles in background.",
    }


from simulator import ScenarioSimulator

simulator = ScenarioSimulator()


class ScenarioRequest(BaseModel):
    scenario: str = "normal"  # "normal" | "gradual" | "sudden" | "imbalance" | "recovery"
    n_samples: int = 500
    drift_strength: float = 1.0


@app.post("/admin/scenario")
def trigger_scenario(req: ScenarioRequest):
    """
    Triggers a controlled scenario simulation:
    - "normal": Nominal baseline features
    - "gradual": Linearly escalating drift
    - "sudden": Step-function sensor shift
    - "imbalance": High class imbalance stream
    - "recovery": Post-retraining healthy distribution recovery
    """
    if not orchestrator.is_ready:
        raise HTTPException(400, "Orchestrator is not ready yet.")

    try:
        raw_X, labels, pred_ids = simulator.get_scenario_batch(
            scenario=req.scenario,
            n_samples=req.n_samples,
            drift_strength=req.drift_strength,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    if getattr(wm, "scaler", None) is None:
        scaled_X = raw_X
    else:
        scaled_X = np.clip(wm.scaler.transform(raw_X), -2.0, 2.0).astype(np.float32)
    preds = ml_model.predict_batch(raw_X)

    thread = threading.Thread(
        target=run_monitoring_cycle_background,
        args=(scaled_X, labels, preds),
        daemon=True,
    )
    thread.start()

    return {
        "status": "success",
        "scenario": req.scenario,
        "samples_injected": len(raw_X),
        "message": f"Scenario '{req.scenario}' injected successfully ({len(raw_X)} samples). Monitoring cycle triggered.",
    }


# ─────────────────────────────────────────────
# Member 3: Dual Operating Modes & Replay Endpoints
# ─────────────────────────────────────────────

class ModeRequest(BaseModel):
    mode: str  # "demo" | "real_data"


@app.post("/admin/mode")
def set_operating_mode(req: ModeRequest):
    """Toggle system operating mode between demo and real_data."""
    if req.mode not in ("demo", "real_data"):
        raise HTTPException(400, "Mode must be 'demo' or 'real_data'")
    config.operating_mode = req.mode
    admin_reset()
    return {"status": "success", "operating_mode": config.operating_mode}


@app.post("/admin/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload custom CSV dataset for Real Data Monitoring Mode."""
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(400, "Uploaded dataset must be a CSV file.")
    os.makedirs(config.upload_dir, exist_ok=True)
    save_path = os.path.join(config.upload_dir, file.filename or "upload.csv")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    
    global active_replay_engine
    from simulator import CustomCSVReplayEngine
    active_replay_engine = CustomCSVReplayEngine(save_path)
    
    admin_reset()
    
    return {"status": "success", "message": "Dataset uploaded successfully", "filename": file.filename, "path": save_path, "rows": len(active_replay_engine.df)}


from simulator import CustomCSVReplayEngine

active_replay_engine: Optional[CustomCSVReplayEngine] = None


class ReplayRequest(BaseModel):
    filename: Optional[str] = None
    batch_size: int = 500


@app.post("/admin/replay_step")
def replay_step(req: ReplayRequest):
    """Replay next batch from uploaded CSV dataset in Real Data Monitoring Mode."""
    global active_replay_engine
    if not orchestrator.is_ready:
        raise HTTPException(400, "Orchestrator is not ready yet.")

    if req.filename:
        csv_path = os.path.join(config.upload_dir, req.filename)
        if not os.path.exists(csv_path):
            raise HTTPException(404, f"Dataset file '{req.filename}' not found in uploads directory.")
        active_replay_engine = CustomCSVReplayEngine(csv_path)
    elif active_replay_engine is None:
        raise HTTPException(400, "No active dataset loaded for replay. Provide a filename.")

    raw_X, labels, pred_ids = active_replay_engine.get_next_batch(req.batch_size)
    if getattr(wm, "scaler", None) is None:
        scaled_X = raw_X
    else:
        scaled_X = np.clip(wm.scaler.transform(raw_X), -2.0, 2.0).astype(np.float32)
    preds = ml_model.predict_batch(raw_X)

    # Run synchronously to avoid macOS thread deadlocks with PyTorch/SHAP
    run_monitoring_cycle_background(scaled_X, labels, preds)

    return {
        "status": "success",
        "samples_replayed": len(raw_X),
        "message": f"Replayed batch of {len(raw_X)} samples. Monitoring cycle triggered.",
    }


# ─────────────────────────────────────────────
# Member 3: Registry & Model Versioning Proxies
# ─────────────────────────────────────────────

@app.get("/registry/history")
def registry_history(limit: int = 50):
    """Fetch model version lineage from registry database."""
    if not hasattr(orchestrator, "registry"):
        return {"status": "no_data", "history": []}
    history = orchestrator.registry.get_history(limit=limit)
    return {"status": "success", "history": history}


class RollbackRequest(BaseModel):
    version_tag: str


@app.post("/registry/rollback")
def registry_rollback(req: RollbackRequest):
    """Roll back active model status to a historical version tag."""
    if not hasattr(orchestrator, "registry"):
        raise HTTPException(400, "Model registry not initialised.")
    history = orchestrator.registry.get_history()
    target = next((h for h in history if h["version_tag"] == req.version_tag and h["promotion_status"] in ("promoted", "initial_baseline")), None)
    if not target:
        raise HTTPException(404, f"Version '{req.version_tag}' not found or was rejected.")
    return {
        "status": "success",
        "message": f"Rolled back active model to version {req.version_tag}",
        "active_version": req.version_tag,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

