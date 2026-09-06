"""
Experiment Tracker — Model Decay Radar
=======================================
Structured SQLite-based experiment logging system that records every
retraining attempt with before/after metrics, training data statistics,
promotion decisions, and timing information for auditability and
presentation purposes.
"""

import os
import sqlite3
import time
import json
from typing import Dict, Any, List, Optional
from config import config


class ExperimentTracker:
    """
    Logs every retrain experiment into a dedicated SQLite table
    with structured before/after metrics and training metadata.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or getattr(config, "experiment_db_path", "logs/experiments.db")
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS retrain_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    classifier_type TEXT NOT NULL,
                    batch_id INTEGER,
                    sample_count INTEGER,
                    reference_samples INTEGER,
                    drift_samples INTEGER,
                    smote_applied INTEGER DEFAULT 0,
                    class_balance_before TEXT,
                    class_balance_after TEXT,
                    before_metrics TEXT,
                    after_metrics TEXT,
                    delta_metrics TEXT,
                    promotion_decision TEXT NOT NULL,
                    rejection_reason TEXT,
                    version_tag TEXT,
                    duration_ms REAL
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def start_experiment(
        self,
        trigger_reason: str,
        classifier_type: str,
        batch_id: int = 0,
    ) -> Dict[str, Any]:
        """Start tracking a new experiment. Returns an experiment context dict."""
        return {
            "experiment_id": f"exp_{int(time.time() * 1000)}_{batch_id}",
            "timestamp": time.time(),
            "trigger_reason": trigger_reason,
            "classifier_type": classifier_type,
            "batch_id": batch_id,
            "_start_time_ns": time.perf_counter_ns(),
        }

    def log_experiment(
        self,
        context: Dict[str, Any],
        sample_count: int,
        reference_samples: int,
        drift_samples: int,
        smote_applied: bool,
        class_balance_before: Dict[int, int],
        class_balance_after: Dict[int, int],
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float],
        promotion_decision: str,
        rejection_reason: Optional[str] = None,
        version_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a completed experiment with all metadata."""
        start_ns = context.get("_start_time_ns", time.perf_counter_ns())
        duration_ms = (time.perf_counter_ns() - start_ns) / 1e6

        # Compute deltas
        delta_metrics = {}
        for key in before_metrics:
            if key in after_metrics:
                delta_metrics[f"{key}_delta"] = round(after_metrics[key] - before_metrics[key], 4)

        record = {
            "experiment_id": context["experiment_id"],
            "timestamp": context["timestamp"],
            "trigger_reason": context["trigger_reason"],
            "classifier_type": context["classifier_type"],
            "batch_id": context.get("batch_id", 0),
            "sample_count": sample_count,
            "reference_samples": reference_samples,
            "drift_samples": drift_samples,
            "smote_applied": smote_applied,
            "class_balance_before": class_balance_before,
            "class_balance_after": class_balance_after,
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
            "delta_metrics": delta_metrics,
            "promotion_decision": promotion_decision,
            "rejection_reason": rejection_reason,
            "version_tag": version_tag,
            "duration_ms": round(duration_ms, 2),
        }

        # Persist to SQLite
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO retrain_experiments (
                    experiment_id, timestamp, trigger_reason, classifier_type,
                    batch_id, sample_count, reference_samples, drift_samples,
                    smote_applied, class_balance_before, class_balance_after,
                    before_metrics, after_metrics, delta_metrics,
                    promotion_decision, rejection_reason, version_tag, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record["experiment_id"],
                record["timestamp"],
                record["trigger_reason"],
                record["classifier_type"],
                record["batch_id"],
                record["sample_count"],
                record["reference_samples"],
                record["drift_samples"],
                int(record["smote_applied"]),
                json.dumps(record["class_balance_before"]),
                json.dumps(record["class_balance_after"]),
                json.dumps(record["before_metrics"]),
                json.dumps(record["after_metrics"]),
                json.dumps(record["delta_metrics"]),
                record["promotion_decision"],
                record["rejection_reason"],
                record["version_tag"],
                record["duration_ms"],
            ))
            conn.commit()
        finally:
            conn.close()

        print(f"[ExperimentTracker] Logged experiment {record['experiment_id']}: "
              f"{record['promotion_decision']} ({record['duration_ms']:.1f}ms)")
        return record

    def get_experiment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent experiment records."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT experiment_id, timestamp, trigger_reason, classifier_type,
                       batch_id, sample_count, reference_samples, drift_samples,
                       smote_applied, class_balance_before, class_balance_after,
                       before_metrics, after_metrics, delta_metrics,
                       promotion_decision, rejection_reason, version_tag, duration_ms
                FROM retrain_experiments
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({
                    "experiment_id": row[0],
                    "timestamp": row[1],
                    "trigger_reason": row[2],
                    "classifier_type": row[3],
                    "batch_id": row[4],
                    "sample_count": row[5],
                    "reference_samples": row[6],
                    "drift_samples": row[7],
                    "smote_applied": bool(row[8]),
                    "class_balance_before": json.loads(row[9]) if row[9] else {},
                    "class_balance_after": json.loads(row[10]) if row[10] else {},
                    "before_metrics": json.loads(row[11]) if row[11] else {},
                    "after_metrics": json.loads(row[12]) if row[12] else {},
                    "delta_metrics": json.loads(row[13]) if row[13] else {},
                    "promotion_decision": row[14],
                    "rejection_reason": row[15],
                    "version_tag": row[16],
                    "duration_ms": row[17],
                })
            return results
        finally:
            conn.close()

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Aggregate summary statistics across all experiments."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM retrain_experiments")
            total = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM retrain_experiments WHERE promotion_decision = 'promoted'")
            promoted = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM retrain_experiments WHERE promotion_decision = 'rejected'")
            rejected = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(duration_ms) FROM retrain_experiments")
            avg_duration = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT COUNT(*) FROM retrain_experiments WHERE smote_applied = 1")
            smote_count = cursor.fetchone()[0]

            return {
                "total_experiments": total,
                "promoted_count": promoted,
                "rejected_count": rejected,
                "promotion_rate": round(promoted / max(total, 1), 4),
                "avg_duration_ms": round(avg_duration, 2),
                "smote_applied_count": smote_count,
            }
        finally:
            conn.close()
