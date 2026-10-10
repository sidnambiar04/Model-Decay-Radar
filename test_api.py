import requests
import time

url = "http://localhost:8000/predict"
payload = {
    "features": {
        "FIT101": 2.45,
        "LIT101": 505.2,
        "DPIT301": 19.8,
        "P402": 1.0
    }
}
try:
    time.sleep(5)  # Wait for API to be up
    res = requests.post(url, json=payload)
    print("STATUS:", res.status_code)
    print("RESPONSE:", res.json())
except Exception as e:
    print("ERROR:", e)
