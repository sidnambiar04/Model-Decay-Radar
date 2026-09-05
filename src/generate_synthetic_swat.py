import numpy as np
import pandas as pd
np.random.seed(42)

SENSOR_TYPES   = ["FIT","LIT","AIT","DPIT","PIT"]
ACTUATOR_TYPES = ["MV","P","UV"]
tags = []
for s in range(1,7):
    for st in SENSOR_TYPES:
        tags.append(f"{st}{s}01")
for s in range(1,7):
    for at in ACTUATOR_TYPES:
        for n in [1,2]:
            tags.append(f"{at}{s}0{n}")
tags = sorted(set(tags))[:51]
for forced in ["LIT101","DPIT301","P402"]:
    if forced not in tags:
        tags[np.random.randint(0,len(tags))] = forced
tags = sorted(set(tags))
while len(tags) < 51: tags.append(f"EXTRA{len(tags):02d}")
tags = tags[:51]

N_REF, N_STABLE, N_DRIFT = 20000, 10000, 8000
N_TOTAL = N_REF + N_STABLE + N_DRIFT
t = np.arange(N_TOTAL)
data = {}
for tag in tags:
    bm = np.random.uniform(20,80)
    bs = np.random.uniform(1,5)
    freq = np.random.uniform(0.001,0.01)
    data[tag] = bm + 5*np.sin(2*np.pi*freq*t) + np.random.normal(0,bs,N_TOTAL)
df = pd.DataFrame(data)

drift_start = N_REF + N_STABLE
for tag in ["LIT101","DPIT301","P402"]:
    bm = df[tag].iloc[:N_REF].mean()
    bs = df[tag].iloc[:N_REF].std()
    drift_len = N_TOTAL - drift_start
    dt = np.arange(drift_len)
    ramp = np.linspace(0,1,drift_len)
    new_freq = np.random.uniform(0.05,0.1)
    regime = bm + ramp*(bs*4) + 8*np.sin(2*np.pi*new_freq*dt)*ramp + np.random.normal(0,bs*(1+2*ramp),drift_len)
    new_sig = df[tag].copy()
    new_sig.iloc[drift_start:] = regime
    df[tag] = new_sig

labels = np.zeros(N_TOTAL, dtype=int)
drift_region = np.arange(drift_start, N_TOTAL)
n_anom = int(len(drift_region)*0.08)
labels[np.random.choice(drift_region,n_anom,replace=False)] = 1
df["Label"] = labels
df["_window"] = (["reference"]*N_REF + ["current_stable"]*N_STABLE + ["current_drift"]*N_DRIFT)

import os
os.makedirs(os.path.join(os.path.dirname(__file__), "../data"), exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "../data/synthetic_swat.csv")
df.to_csv(out, index=False)
print(f"Saved {df.shape} to {out}")
print(f"Label dist: {df['Label'].value_counts().to_dict()}")
