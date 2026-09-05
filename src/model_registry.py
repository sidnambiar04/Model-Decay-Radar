"""
SQLite Model Registry & Version Tracking — Model Decay Radar
============================================================
Lightweight SQLite storage for model versions, validation metrics,
promotion status (promoted vs rejected), and drift event lineage.
"""

import os
import sqlite3
import time
from typing import Dict, Any, List, Optional
from config import config


class ModelRegistry:
    """
    Manages lightweight SQLite database for model version tracking (v1, v2, v3).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "db_path", "models/model_registry.db")
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    classifier_type TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    promotion_status TEXT NOT NULL,
                    drift_event_id INTEGER,
                    rejection_reason TEXT
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def get_latest_version_tag(self) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_tag FROM model_versions
                WHERE promotion_status = 'promoted' OR promotion_status = 'initial_baseline'
                ORDER BY id DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                return row[0]
            return "v1"
        finally:
            conn.close()

    def get_next_version_tag(self) -> str:
        latest = self.get_latest_version_tag()
        if latest.startswith("v") and latest[1:].isdigit():
            v_num = int(latest[1:]) + 1
            return f"v{v_num}"
        return "v1"

    def log_version(
        self,
        version_tag: str,
        classifier_type: str,
        metrics: Dict[str, float],
        promotion_status: str,
        drift_event_id: Optional[int] = None,
        rejection_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        ts = time.time()
        acc = metrics.get("accuracy", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        f1 = metrics.get("f1_score", 0.0)

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_versions (
                    version_tag, timestamp, classifier_type,
                    accuracy, precision, recall, f1_score,
                    promotion_status, drift_event_id, rejection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (version_tag, ts, classifier_type, acc, prec, rec, f1, promotion_status, drift_event_id, rejection_reason))
            conn.commit()
        finally:
            conn.close()

        return {
            "version_tag": version_tag,
            "timestamp": ts,
            "classifier_type": classifier_type,
            "promotion_status": promotion_status,
            "metrics": metrics,
        }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_tag, timestamp, classifier_type,
                       accuracy, precision, recall, f1_score,
                       promotion_status, drift_event_id, rejection_reason
                FROM model_versions
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "version_tag": row[0],
                    "timestamp": row[1],
                    "classifier_type": row[2],
                    "accuracy": row[3],
                    "precision": row[4],
                    "recall": row[5],
                    "f1_score": row[6],
                    "promotion_status": row[7],
                    "drift_event_id": row[8],
                    "rejection_reason": row[9],
                })
            return results
        finally:
            conn.close()
