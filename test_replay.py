import sys
import threading
import time
import numpy as np

from api.server import *
import requests

try:
    print("Setting up orchestrator...")
    orchestrator.setup(reference_scaled, reference_y, rnn_epochs=1)
    
    print("Testing get_next_batch...")
    active_replay_engine = CustomCSVReplayEngine("data/synthetic_swat.csv")
    raw_X, labels, pred_ids = active_replay_engine.get_next_batch(500)
    scaled_X = np.clip(wm.scaler.transform(raw_X), -2.0, 2.0).astype(np.float32)
    preds = ml_model.predict_batch(raw_X)
    
    print("Running background cycle synchronously to see errors...")
    run_monitoring_cycle_background(scaled_X, labels, preds)
except Exception as e:
    import traceback
    traceback.print_exc()

