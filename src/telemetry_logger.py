"""
Telemetry Logger — PostgreSQL Database Persistence Module
==========================================================
Manages logging and retrieval of MonitoringResult telemetry records
to PostgreSQL (table: telemetry_history).
"""

import json
import logging
from typing import List, Dict, Any, Optional

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger("TelemetryLogger")


class TelemetryLogger:
    def __init__(self, postgres_url: str = "postgresql://postgres:postgres@localhost:5432/model_decay_radar"):
        self.postgres_url = postgres_url
        self.is_connected = False
        self.fallback_history: List[Dict[str, Any]] = []
        self._init_db()

    def _get_connection(self):
        if not PSYCOPG2_AVAILABLE:
            return None
        try:
            conn = psycopg2.connect(self.postgres_url, connect_timeout=3)
            return conn
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}. Using fallback telemetry store.")
            return None

    def _init_db(self):
        """Create telemetry_history table in PostgreSQL if available."""
        conn = self._get_connection()
        if not conn:
            self.is_connected = False
            return

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS telemetry_history (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        batch_id INT,
                        model_version VARCHAR(50),
                        model_health_score FLOAT,
                        drift_severity FLOAT,
                        kl_p_value FLOAT,
                        uncertainty_score FLOAT,
                        status VARCHAR(50),
                        drift_category VARCHAR(50),
                        raw_payload JSONB
                    );
                """)
                conn.commit()
            self.is_connected = True
            logger.info("Successfully connected to PostgreSQL telemetry database.")
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL schema: {e}")
            self.is_connected = False
        finally:
            conn.close()

    def log_monitoring_result(self, result: Any) -> bool:
        """Persist a MonitoringResult dataclass to PostgreSQL (or fallback)."""
        # Convert MonitoringResult or dict to JSON serializable dictionary
        if hasattr(result, "to_dict"):
            data = result.to_dict()
        elif isinstance(result, dict):
            data = result
        else:
            data = {"raw": str(result)}

        # Append to memory fallback history
        self.fallback_history.append(data)
        if len(self.fallback_history) > 500:
            self.fallback_history.pop(0)

        conn = self._get_connection()
        if not conn:
            return False

        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO telemetry_history (
                        batch_id, model_version, model_health_score, drift_severity,
                        kl_p_value, uncertainty_score, status, drift_category, raw_payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    data.get("batch_id", 0),
                    data.get("model_version", "v1"),
                    data.get("model_health_score", 1.0),
                    data.get("drift_severity", 0.0),
                    data.get("kl_p_value", 1.0),
                    data.get("uncertainty_score", 0.0),
                    data.get("status", "Healthy"),
                    data.get("drift_category", "none"),
                    json.dumps(data)
                ))
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to log telemetry result to PostgreSQL: {e}")
            return False
        finally:
            conn.close()

    def get_recent_history(self, limit: int = 200) -> List[Dict[str, Any]]:
        """Fetch recent telemetry history from PostgreSQL or memory fallback."""
        conn = self._get_connection()
        if not conn:
            return self.fallback_history[-limit:]

        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT raw_payload FROM telemetry_history
                    ORDER BY id DESC LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
                results = [row["raw_payload"] for row in reversed(rows)]
                return results if results else self.fallback_history[-limit:]
        except Exception as e:
            logger.error(f"Failed to fetch telemetry history from PostgreSQL: {e}")
            return self.fallback_history[-limit:]
        finally:
            conn.close()

    def clear_history(self):
        """Clear telemetry history from both PostgreSQL and memory fallback."""
        self.fallback_history.clear()
        conn = self._get_connection()
        if conn:
            try:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE telemetry_history;")
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to clear telemetry history from PostgreSQL: {e}")
            finally:
                conn.close()
