"use client";

import { useState, useEffect, useCallback } from "react";

export interface DriftFeature {
  feature: string;
  importance: number;
}

export interface DriftFeatureKS {
  feature: string;
  importance: number; // ks_stat
  p_value: number;
  drift_confirmed: boolean;
}

export interface MonitoringResult {
  batch_id: number;
  timestamp: number;
  mean_reconstruction_error: number;
  dynamic_threshold: number;
  observed_kl: number;
  p_value: number;
  drift_confirmed: boolean;
  drift_severity: number;
  wasserstein_distance: number;
  score_ks_p_value: number;
  mean_uncertainty: number;
  mhs: number;
  mhs_status: "Healthy" | "Warning" | "Critical" | "Initialising";
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  top_drift_features: DriftFeature[];
  feature_ks_results: DriftFeatureKS[];
  smote_applied: boolean;
  samples_before_smote: number;
  samples_after_smote: number;
  alert_level: "none" | "warning" | "critical";
  alert_message: string;
  retraining_triggered: boolean;
}

export interface HistoryResponse {
  results: MonitoringResult[];
  total: number;
}

export interface HealthStatus {
  status: string;
  orchestrator_ready: boolean;
  setup_progress: string;         // "pending" | "training_ae" | "training_rnn" | "ready" | "failed"
  setup_error: string | null;
  buffer_size: number;
  window_size: number;
  monitoring_cycles_completed: number;
}

export interface ServerConfig {
  active_classifier: string;
  available_classifiers: string[];
  window_size: number;
  p_value_threshold: number;
}

const API_BASE = "http://localhost:8000";

export function useMonitoring() {
  const [latest, setLatest] = useState<MonitoringResult | null>(null);
  const [history, setHistory] = useState<MonitoringResult[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [config, setConfig] = useState<ServerConfig | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [simulateMessage, setSimulateMessage] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [healthRes, latestRes, historyRes, configRes] = await Promise.all([
        fetch(`${API_BASE}/health`),
        fetch(`${API_BASE}/monitoring/latest`),
        fetch(`${API_BASE}/monitoring/history?limit=50`),
        fetch(`${API_BASE}/admin/config`),
      ]);

      if (!healthRes.ok || !latestRes.ok || !historyRes.ok || !configRes.ok) {
        throw new Error("Failed to fetch data from server");
      }

      const healthData: HealthStatus = await healthRes.json();
      const latestData = await latestRes.json();
      const historyData: HistoryResponse = await historyRes.json();
      const configData: ServerConfig = await configRes.json();

      setHealth(healthData);
      setConfig(configData);
      setIsConnected(true);
      setError(null);

      if (latestData.status === "no_data") {
        setLatest(null);
      } else {
        setLatest(latestData);
      }

      setHistory(historyData.results || []);
    } catch (err: any) {
      setIsConnected(false);
      setError(err.message || "Failed to connect to monitoring server");
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 3000);
    return () => clearInterval(interval);
  }, [fetchData]);

  const simulateDrift = async (): Promise<string> => {
    if (!health?.orchestrator_ready) {
      const msg =
        "⏳ Orchestrator is still training (AE + RNN). Please wait until training completes (~1-2 min after server start).";
      setSimulateMessage(msg);
      return msg;
    }

    try {
      const res = await fetch(`${API_BASE}/admin/simulate`);
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const data = await res.json();
      const msg = `✅ ${data.message}`;
      setSimulateMessage(msg);
      // Start polling faster briefly so the user sees results quickly
      setTimeout(() => fetchData(), 1000);
      setTimeout(() => fetchData(), 3000);
      setTimeout(() => fetchData(), 6000);
      setTimeout(() => fetchData(), 10000);
      setTimeout(() => fetchData(), 15000);
      return msg;
    } catch (err: any) {
      const msg = `❌ Simulation failed: ${err.message}`;
      setSimulateMessage(msg);
      return msg;
    }
  };

  const resetDashboard = async () => {
    try {
      const res = await fetch(`${API_BASE}/admin/reset`, { method: "POST" });
      if (!res.ok) {
        throw new Error("Failed to reset");
      }
      setLatest(null);
      setHistory([]);
      setSimulateMessage(null);
      await fetchData();
    } catch (err: any) {
      console.error(err);
      alert(err.message || "Error resetting data");
    }
  };

  const updateConfig = async (newConfig: { active_classifier?: string; window_size?: number; p_value_threshold?: number }): Promise<boolean> => {
    try {
      const res = await fetch(`${API_BASE}/admin/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newConfig),
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "Failed to update configuration");
      }
      await fetchData();
      return true;
    } catch (err: any) {
      alert(err.message || "Failed to update configuration");
      return false;
    }
  };

  const triggerManualRetrain = async (): Promise<string> => {
    try {
      const res = await fetch(`${API_BASE}/admin/retrain`, { method: "POST" });
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      const data = await res.json();
      await fetchData();
      return `✅ ${data.message}`;
    } catch (err: any) {
      return `❌ Retraining failed: ${err.message}`;
    }
  };

  return {
    latest,
    history,
    health,
    config,
    isConnected,
    isLoading,
    error,
    simulateMessage,
    simulateDrift,
    resetDashboard,
    updateConfig,
    triggerManualRetrain,
    refetch: fetchData,
  };
}

