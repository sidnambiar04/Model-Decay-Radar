import os
import time
import asyncio
import httpx
import random
from typing import Dict, Any

MDR_INGEST_URL = os.getenv("MDR_INGEST_URL", "http://localhost:8000/ingest")
API_KEY = os.getenv("MDR_API_KEY", "dev-test-key-123")

class MDRClient:
    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key
        self.queue = asyncio.Queue()
        self.worker_task = None
        
    async def start(self):
        self.worker_task = asyncio.create_task(self._worker())
        print("[MDRClient] Background ingestion worker started.")
        
    async def _worker(self):
        async with httpx.AsyncClient(timeout=3.0) as client:
            while True:
                payload = await self.queue.get()
                try:
                    headers = {"X-API-Key": self.api_key}
                    # Send fire-and-forget webhook
                    await client.post(self.url, json=payload, headers=headers)
                except Exception as e:
                    print(f"[MDRClient] Failed to send telemetry: {e}")
                finally:
                    self.queue.task_done()
                    
    def log_prediction(self, features: Dict[str, float], prediction: float, label: int = None, model_id: str = "my_custom_model"):
        """Non-blocking log to queue"""
        payload = {
            "model_id": model_id,
            "features": features,
            "prediction": prediction,
            "label": label
        }
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            print("[MDRClient] Queue full, dropping telemetry to prevent memory leak.")


# --- Example Usage ---

async def main():
    mdr = MDRClient(MDR_INGEST_URL, API_KEY)
    await mdr.start()
    
    print("Simulating User's production server receiving requests...")
    
    for i in range(50):
        # 1. User receives data from IoT device
        sensor_data = {
            "FIT101": 2.5 + random.uniform(-0.1, 0.1),
            "LIT101": 500.0 + random.uniform(-10, 10),
        }
        
        # 2. User's model makes prediction instantly
        prediction = 0 if sum(sensor_data.values()) < 505 else 1
        
        # 3. User asynchronously logs to MDR webhook (non-blocking)
        mdr.log_prediction(features=sensor_data, prediction=prediction, model_id="pump_monitor_v2")
        
        # 4. User responds to their client instantly
        # print(f"Served prediction {prediction} to client.")
        
        await asyncio.sleep(0.01) # Simulating traffic
        
    print("Finished serving requests. Waiting for webhook queue to flush...")
    await mdr.queue.join()
    print("All telemetry synced!")

if __name__ == "__main__":
    asyncio.run(main())
