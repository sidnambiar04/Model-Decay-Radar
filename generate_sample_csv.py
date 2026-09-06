import pandas as pd
from src.simulator import SWaTSimulator

print("Generating 1000 samples using SWaTSimulator...")
sim = SWaTSimulator()
# Generate 500 normal samples
X_normal, y_normal = sim.generate_batch(500, drift_scenario="normal")
df_normal = pd.DataFrame(X_normal, columns=sim.feature_names)
df_normal["Normal/Attack"] = ["Normal" if y == 0 else "Attack" for y in y_normal]

# Generate 500 drifted samples (e.g. sensor drift)
X_drift, y_drift = sim.generate_batch(500, drift_scenario="sensor_drift", drift_strength=5.0)
df_drift = pd.DataFrame(X_drift, columns=sim.feature_names)
df_drift["Normal/Attack"] = ["Normal" if y == 0 else "Attack" for y in y_drift]

df_combined = pd.concat([df_normal, df_drift], ignore_index=True)
df_combined.to_csv("sample_real_data.csv", index=False)
print("Successfully created sample_real_data.csv in the current directory!")
