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
                    rejection_reason TEXT,
                    artifact_path TEXT
                )
            """)
            conn.commit()

            # Auto-migrate schema if artifact_path column is missing in existing table
            cursor.execute("PRAGMA table_info(model_versions)")
            columns = [col[1] for col in cursor.fetchall()]
            if "artifact_path" not in columns:
                print("[ModelRegistry] Auto-migrating schema: Adding 'artifact_path' column to model_versions table.")
                cursor.execute("ALTER TABLE model_versions ADD COLUMN artifact_path TEXT")
                conn.commit()

            # Training lineage table — tracks dataset composition for each version
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_lineage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version_tag TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    reference_samples INTEGER,
                    drift_samples INTEGER,
                    total_samples INTEGER,
                    smote_applied INTEGER DEFAULT 0,
                    imbalance_ratio_before REAL,
                    imbalance_ratio_after REAL,
                    training_duration_ms REAL,
                    drift_event_batch_id INTEGER,
                    notes TEXT
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
        artifact_path: Optional[str] = None,
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
                    promotion_status, drift_event_id, rejection_reason, artifact_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (version_tag, ts, classifier_type, acc, prec, rec, f1, promotion_status, drift_event_id, rejection_reason, artifact_path))
            conn.commit()
        finally:
            conn.close()

        return {
            "version_tag": version_tag,
            "timestamp": ts,
            "classifier_type": classifier_type,
            "promotion_status": promotion_status,
            "metrics": metrics,
            "artifact_path": artifact_path,
        }

    def get_version_details(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """Fetch detailed version metadata for a specific version tag."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_tag, timestamp, classifier_type,
                       accuracy, precision, recall, f1_score,
                       promotion_status, drift_event_id, rejection_reason, artifact_path
                FROM model_versions
                WHERE version_tag = ?
                ORDER BY id DESC LIMIT 1
            """, (version_tag,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
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
                "artifact_path": row[10],
            }
        finally:
            conn.close()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_tag, timestamp, classifier_type,
                       accuracy, precision, recall, f1_score,
                       promotion_status, drift_event_id, rejection_reason, artifact_path
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
                    "artifact_path": row[10],
                })
            return results
        finally:
            conn.close()

    def rollback_to_version(self, target_version_tag: str, classifier_engine: Optional[Any] = None) -> Dict[str, Any]:
        """
        Rollback active production model to a previously promoted historical version tag.
        Updates model registry logs and optionally loads binary model weights into classifier_engine.
        """
        details = self.get_version_details(target_version_tag)
        if not details:
            raise ValueError(f"Cannot rollback: Version tag '{target_version_tag}' not found in registry.")

        if details["promotion_status"] not in ("promoted", "initial_baseline"):
            raise ValueError(f"Cannot rollback to version '{target_version_tag}' with status '{details['promotion_status']}'. Only promoted models can be restored.")

        artifact_path = details.get("artifact_path")
        if classifier_engine and artifact_path and os.path.exists(artifact_path):
            classifier_engine.load_model_artifact(artifact_path)

        # Log rollback event as active baseline
        rollback_metrics = {
            "accuracy": details.get("accuracy", 0.0),
            "precision": details.get("precision", 0.0),
            "recall": details.get("recall", 0.0),
            "f1_score": details.get("f1_score", 0.0),
        }
        return self.log_version(
            version_tag=target_version_tag,
            classifier_type=details["classifier_type"],
            metrics=rollback_metrics,
            promotion_status="rollback_promoted",
            rejection_reason=f"Manual rollback restored to version {target_version_tag}.",
            artifact_path=artifact_path,
        )

    def log_training_lineage(
        self,
        version_tag: str,
        reference_samples: int,
        drift_samples: int,
        smote_applied: bool = False,
        imbalance_ratio_before: float = 1.0,
        imbalance_ratio_after: float = 1.0,
        training_duration_ms: float = 0.0,
        drift_event_batch_id: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log training data lineage for a model version."""
        ts = time.time()
        total = reference_samples + drift_samples

        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_lineage (
                    version_tag, timestamp, reference_samples, drift_samples,
                    total_samples, smote_applied, imbalance_ratio_before,
                    imbalance_ratio_after, training_duration_ms,
                    drift_event_batch_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version_tag, ts, reference_samples, drift_samples,
                total, int(smote_applied), imbalance_ratio_before,
                imbalance_ratio_after, training_duration_ms,
                drift_event_batch_id, notes,
            ))
            conn.commit()
        finally:
            conn.close()

        return {
            "version_tag": version_tag,
            "timestamp": ts,
            "reference_samples": reference_samples,
            "drift_samples": drift_samples,
            "total_samples": total,
            "smote_applied": smote_applied,
            "imbalance_ratio_before": imbalance_ratio_before,
            "imbalance_ratio_after": imbalance_ratio_after,
            "training_duration_ms": training_duration_ms,
            "drift_event_batch_id": drift_event_batch_id,
            "notes": notes,
        }

    def get_training_lineage(self, version_tag: str) -> Optional[Dict[str, Any]]:
        """Retrieve training lineage metadata for a specific model version."""
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT version_tag, timestamp, reference_samples, drift_samples,
                       total_samples, smote_applied, imbalance_ratio_before,
                       imbalance_ratio_after, training_duration_ms,
                       drift_event_batch_id, notes
                FROM training_lineage
                WHERE version_tag = ?
                ORDER BY id DESC LIMIT 1
            """, (version_tag,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "version_tag": row[0],
                "timestamp": row[1],
                "reference_samples": row[2],
                "drift_samples": row[3],
                "total_samples": row[4],
                "smote_applied": bool(row[5]),
                "imbalance_ratio_before": row[6],
                "imbalance_ratio_after": row[7],
                "training_duration_ms": row[8],
                "drift_event_batch_id": row[9],
                "notes": row[10],
            }
        finally:
            conn.close()

