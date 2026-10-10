import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))

from celery_app import celery_app
from config import config
from pipeline.orchestrator import RadarOrchestrator
from data_pipeline import DataWindowManager
from ai_assistant import generate_drift_explanation
from telemetry_logger import TelemetryLogger

# Global singletons for the worker process
telemetry_logger = TelemetryLogger(postgres_url=config.postgres_url)
orchestrator: Optional[RadarOrchestrator] = None
wm: Optional[DataWindowManager] = None
FEATURE_COLS = []

def get_orchestrator():
    global orchestrator, wm, FEATURE_COLS
    if orchestrator is not None and orchestrator.is_ready:
        return orchestrator, wm, FEATURE_COLS

    print("[Worker] Initializing Radar Orchestrator...", flush=True)
    DATA_PATH = os.path.join(_ROOT, "data", "synthetic_swat.csv")
    df = pd.read_csv(DATA_PATH)
    FEATURE_COLS = [c for c in df.columns if c not in ("Label", "_window")]
    
    wm = DataWindowManager(df, FEATURE_COLS, label_col="Label")
    reference_df = df[df["_window"] == "reference"]
    wm.set_reference_window(reference_df)
    
    orchestrator = RadarOrchestrator(feature_names=FEATURE_COLS)
    
    # We do NOT pass a classifier, because in MaaS mode, the user owns the model.
    orchestrator.setup(
        reference_scaled=wm.reference_scaled,
        reference_raw_X=reference_df[FEATURE_COLS].to_numpy(dtype=np.float32),
        reference_raw_y=reference_df["Label"].to_numpy(dtype=np.int32),
        classifier=None,  # Disabled retraining for external models
        scaler=wm.scaler,
        ae_epochs=config.ae_epochs,
        rnn_epochs=config.rnn_epochs,
    )
    return orchestrator, wm, FEATURE_COLS

@celery_app.task(name="worker.process_monitoring_window")
def process_monitoring_window(batch_data: List[dict]):
    """
    Celery task that receives a batch of buffered requests from Redis
    and runs the Orchestrator drift detection pipeline.
    """
    try:
        orch, window_manager, feature_names = get_orchestrator()
    except Exception as e:
        print(f"[Worker] Failed to initialize orchestrator: {e}", flush=True)
        return {"status": "error", "message": "Orchestrator setup failed"}

    # Extract data from the batch
    raw_features = []
    predictions = []
    labels = []
    has_labels = False

    for item in batch_data:
        # Assuming features are provided in a dict mapping name -> value
        feats_dict = item.get("features", {})
        row = [feats_dict.get(col, 0.0) for col in feature_names]
        raw_features.append(row)
        predictions.append(item.get("prediction", 0))
        if "label" in item and item["label"] is not None:
            labels.append(item["label"])
            has_labels = True

    raw_features_np = np.array(raw_features, dtype=np.float32)
    predictions_np = np.array(predictions, dtype=np.int32)
    labels_np = np.array(labels, dtype=np.int32) if has_labels and len(labels) == len(raw_features) else None

    # Scale using reference-window scaler
    if window_manager.scaler is None:
        scaled_features = raw_features_np
    else:
        scaled_features = window_manager.scaler.transform(raw_features_np)
        
    scaled_features = np.clip(scaled_features, -2.0, 2.0).astype(np.float32)

    try:
        result = orch.run_monitoring_cycle(
            scaled_features, labels_np, predictions_np
        )
        import dataclasses
        result_dict = dataclasses.asdict(result)
        
        # Trigger Groq AI if drift is detected
        ai_explanation = None
        if result.alert_level in ["warning", "critical"]:
            top_feats = {f["feature"]: f["importance"] for f in result.top_drift_features}
            metrics = {
                "top_drifting_features": top_feats,
                "p_value": result.p_value,
                "wasserstein": result.wasserstein_distance
            }
            ai_explanation = generate_drift_explanation(
                mhs=result.mhs,
                drift_confirmed=result.drift_confirmed,
                alert_level=result.alert_level,
                metrics=metrics
            )
            result_dict["ai_explanation"] = ai_explanation
            print(f"\n[AI Insight] {ai_explanation}\n", flush=True)

        telemetry_logger.log_monitoring_result(result_dict)
        
        print(f"[Worker] Batch {result.batch_id} processed. MHS={result.mhs:.3f}, Drift={result.drift_confirmed}")
        return {"status": "success", "batch_id": result.batch_id, "mhs": result.mhs}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
