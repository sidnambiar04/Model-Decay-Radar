"""
Example Client — how a user / IoT device calls Model Decay Radar
================================================================
This is what runs on the USER SIDE (e.g. a sensor gateway, a webapp,
a data pipeline). It sends feature data to the FastAPI /predict endpoint
and gets back a prediction + the current model health status.

Model Decay Radar does all the monitoring invisibly in the background.
The user just makes a normal predict call — they never have to think
about drift.

Run:
  python example_client.py
"""

import httpx
import pandas as pd
import time
import random

API_URL = "http://localhost:8000"

# Load some sample data (simulating IoT sensor readings)
import os
_ROOT = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(_ROOT, "data/synthetic_swat.csv"))
feature_cols = [c for c in df.columns if c not in ("Label", "_window")]

# Pick samples from the drift window (simulates real production data with drift)
drift_samples = df[df["_window"] == "current_drift"].reset_index(drop=True)

print("=" * 60)
print("Model Decay Radar — Example Client")
print("Sending IoT sensor data to /predict endpoint...")
print("=" * 60)
print()

for i in range(50):
    # Pick a random row from the drift window
    row = drift_samples.iloc[i % len(drift_samples)]

    # Build the features dict (what a real IoT device would send)
    features = {col: float(row[col]) for col in feature_cols}
    label    = int(row["Label"])     # in production: arrives later / not at all

    # Send to FastAPI
    try:
        response = httpx.post(
            f"{API_URL}/predict",
            json={"features": features, "label": label},
            timeout=5.0,
        )

        if response.status_code == 200:
            result = response.json()
            print(f"Sample {i+1:3d} | "
                  f"Prediction: {result['prediction']:.1f} | "
                  f"Confidence: {result['confidence']:.2f} | "
                  f"Model Status: {result['model_status']:10s} | "
                  f"Buffer: {result['batch_buffered']}/500")
        else:
            print(f"Sample {i+1}: Error {response.status_code}")

    except Exception as e:
        print(f"Sample {i+1}: Connection error — is the server running? ({e})")
        break

    time.sleep(0.05)   # 50ms between requests (simulating real sensor rate)

print()
print("=" * 60)
print("Done. Check the dashboard at http://localhost:8501")
print("=" * 60)
