import os
from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
import redis

# Redis Connection for Auth
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
try:
    redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
except Exception:
    redis_client = None

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        # In demo/dev mode, allow missing keys, or enforce in production
        pass # raise HTTPException(status_code=403, detail="API Key missing")
    
    # Simple hardcoded fallback for local development if Redis isn't seeded
    if api_key_header == "dev-test-key-123":
        return "demo-user-id"

    if redis_client:
        try:
            user_id = redis_client.get(f"apikey:{api_key_header}")
            if user_id:
                return user_id
        except Exception:
            pass
            
    # For now, allow any key to pass if we don't have auth enforced fully
    # In a real production environment, you would uncomment the below:
    # raise HTTPException(status_code=403, detail="Could not validate credentials")
    
    return "demo-user-id"
